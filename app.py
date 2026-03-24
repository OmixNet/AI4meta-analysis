import io
import json
import logging
import queue
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    APIConnectionError = Exception
    APITimeoutError = Exception
    InternalServerError = Exception
    RateLimitError = Exception
    OpenAI = None

try:
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    retry = None
    retry_if_exception = None
    stop_after_attempt = None
    wait_exponential = None


DEFAULT_TOPIC = (
    "探讨依靠标志物重新定义肥胖的方法，旨在更准确地筛选肥胖患者，"
    "并实现脂肪病变的更早发现及特征分析。"
)
DEFAULT_BASE_URL = "https://api.openai.com/v1"
SCREENING_TIMEOUT_SECONDS = 60.0
EVALUATOR_TIMEOUT_SECONDS = 90.0
AGGREGATOR_TIMEOUT_SECONDS = 60.0
FORMAT_RETRY_TIMEOUT_SECONDS = 45.0
PDF_LLM_EXTRACTION_TIMEOUT_SECONDS = 75.0
SCREENING_MAX_TOKENS = 300
EVALUATOR_MAX_TOKENS = 1800
AGGREGATOR_MAX_TOKENS = 300
FORMAT_RETRY_MAX_TOKENS = 1200
PDF_LLM_EXTRACTION_MAX_TOKENS = 2800
FORMAT_RETRY_EVIDENCE_MAX_CHARS = 12000
PDF_LLM_EXTRACTION_CHUNK_CHARS = 16000
PDF_LLM_EXTRACTION_SOURCE_MAX_CHARS = 22000
FULLTEXT_UPLOAD_TYPES = ["pdf", "md", "txt"]
FULLTEXT_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
ERROR_RESULT = {
    "score": 0,
    "relevance": "None",
    "decision": "Error",
    "rationale": "API请求失败或超时",
    "error": "API请求失败或超时",
}
VALID_RELEVANCE = {"High", "Medium", "Low", "None"}
VALID_DECISION = {"Include", "Unsure", "Exclude", "Error"}
RELEVANCE_ALIASES = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "none": "None",
}
DECISION_ALIASES = {
    "include": "Include",
    "unsure": "Unsure",
    "exclude": "Exclude",
    "error": "Error",
}
ROB_EVALUATOR_DOMAINS = ["D1", "D2", "D3", "D4", "D5"]
ROB_DOMAINS = [*ROB_EVALUATOR_DOMAINS, "Overall"]
ROB_OUTPUT_COLUMNS = ["Study", *ROB_DOMAINS]
ROB_ALLOWED_SCORES = {"Low", "Some concerns", "High"}
ROB_SCORE_ALIASES = {
    "low": "Low",
    "high": "High",
    "some concerns": "Some concerns",
    "some concern": "Some concerns",
    "some_concerns": "Some concerns",
    "some-concerns": "Some concerns",
}
EVALUATOR_TEMPERATURE = 0.0
AGGREGATOR_TEMPERATURE = 0.0
JsonNormalizer = Callable[[dict[str, Any]], dict[str, Any]]
APP_LOG_PATH = Path(__file__).with_name("rob2_debug.log")
ROB_EVIDENCE_CHUNK_SIZE = 2600
ROB_EVIDENCE_CHUNK_OVERLAP = 300
ROB_EVIDENCE_MAX_CHARS = 22000
ROB_EVIDENCE_KEYWORDS = {
    "random": 5,
    "allocation": 5,
    "conceal": 5,
    "baseline": 4,
    "blind": 5,
    "mask": 4,
    "deviation": 4,
    "adherence": 4,
    "protocol": 5,
    "analysis": 4,
    "intention-to-treat": 6,
    "itt": 4,
    "per-protocol": 4,
    "as-treated": 4,
    "missing": 5,
    "dropout": 5,
    "lost to follow-up": 6,
    "follow-up": 4,
    "outcome": 3,
    "assessor": 4,
    "measurement": 4,
    "registry": 5,
    "registration": 5,
    "statistical": 3,
    "supplement": 2,
    "consort": 4,
}

PDF_EXTRACTION_SYSTEM_PROMPT = """
你是一位专业医学论文 PDF 内容重建专家。你的任务不是总结，而是根据给定的论文 PDF 原始解析文本，重建一份更干净、更准确的正文与表格文本。

执行要求：
- 只保留论文的正文内容、标题、小节标题、方法、结果、讨论、结论以及专业表格数据。
- 删除页眉、页脚、页码、重复行、断裂片段、孤立标点、乱码、装饰符号。
- 不要编造、不要补充、不要总结、不要解释，只能基于输入重组内容。
- 表格数据必须尽量保留，表格每一行使用制表符分隔单元格。
- 保留专业数值、百分比、比较符号、加减号、单位等关键信息。
- 普通标点噪声不要保留，但不能破坏医学含义和表格数值。
- 输出必须是单个 JSON 对象，不要输出任何额外说明。

请严格输出：
{
  "clean_text": "仅包含清洗后的正文与表格文本"
}
""".strip()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rob2_app")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger()

ROB2_GUIDANCE_SUMMARY = """
Apply RoB 2 for individually randomized parallel-group trials and assess one trial result for one outcome.
Use the Cochrane RoB 2 logic internally, including signalling-question style reasoning (Yes / Probably yes / Probably no / No / No information), but output only the requested JSON.

General rules:
- Judge risk of material bias, not trivial reporting imperfections.
- Prefer direct evidence from the article. If information is missing, say that explicitly in the reason.
- Do not assume ideal methods just because the study is called randomized.
- For Domain 2, use the effect of assignment to intervention as the default effect of interest. Treat ITT or modified ITT excluding only missing outcome data as generally appropriate. Treat naive per-protocol or as-treated analyses as inappropriate for this estimand.

Domain anchors:
- D1 Randomization process: Was sequence generation random? Was allocation concealed until enrolment? Do baseline imbalances suggest a randomization problem?
- D2 Deviations from intended interventions: Were participants/personnel aware of assignment? Did trial-context deviations from intended intervention occur? Were deviations likely to affect the outcome and unbalanced? Was analysis appropriate for effect of assignment?
- D3 Missing outcome data: Were outcome data available for all or nearly all randomized participants? Is there evidence the result was not biased by missing data? Could or likely did missingness depend on the true outcome value?
- D4 Measurement of the outcome: Was the measurement method inappropriate? Could measurement/ascertainment differ between groups? Were outcome assessors aware of intervention received? Could or likely did that knowledge influence outcome assessment? Participant-reported outcomes are more vulnerable when unblinded; objective outcomes like all-cause mortality are less vulnerable.
- D5 Selection of the reported result: Was there a pre-specified analysis plan finalized before unblinded outcome data were available? Was the reported result likely selected from multiple eligible measurements, time points, definitions, or analyses?

Overall judgement:
- Low only if all domains are Low.
- High if any domain is High, or if multiple domains raise Some concerns that substantially lower confidence in the result.
- Otherwise Overall is Some concerns.
""".strip()

ROB2_STRICT_WORKFLOW = """
内部执行流程必须固定为以下 4 步，不能跳步：
1. 逐域定位证据：按 D1-D5 顺序在全文中寻找直接证据或明确缺失信息。
2. 逐域判定风险：根据该域的关键问题判断是 Low / Some concerns / High。
3. 压缩理由：把最关键的原文证据或缺失信息压缩为不超过 50 字的理由。
4. 总体裁决：仅在完成 D1-D5 后，严格依 Overall 规则计算 Overall。

提交前必须自检：
- 12 个键全部存在，不能缺少、不能留空、不能写 null。
- 每个 score 只能是 Low / Some concerns / High。
- 如果证据不足，必须给出 Some concerns 或最符合规则的等级，并在 reason 中明确写“未报告”或“信息不足”。
- 禁止输出套话、客套话、模糊评价；你的职责是像严苛同行评审一样挑出偏倚风险。
""".strip()

ROB2_DOMAIN_RUBRIC = """
请把 D1-D5 当作 5 个必须逐项完成的评分任务：
- D1 随机化过程：检查随机序列、分配隐藏、基线失衡。若随机化或隐藏证据充分且无异常，倾向 Low；若信息缺失，倾向 Some concerns；若存在明显非随机或基线失衡异常，倾向 High。
- D2 偏离既定干预：检查盲法、方案偏离、是否采用 ITT 分析评估 assignment effect。若盲法/偏离控制充分且分析恰当，倾向 Low；若有疑点但证据不充分，倾向 Some concerns；若存在明显不当偏离或错误分析，倾向 High。
- D3 结局数据缺失：检查失访比例、组间平衡、缺失原因、插补分析。若缺失极少或充分证明不影响结果，倾向 Low；若缺失信息不足或可能有影响，倾向 Some concerns；若缺失明显且可能改变结论，倾向 High。
- D4 结局测量：检查结局评估者盲法、测量方法客观性、一致性。若测量客观且组间一致，倾向 Low；若主观结局且盲法/测量信息不足，倾向 Some concerns；若测量明显受知晓分组影响，倾向 High。
- D5 选择性报告：检查预注册、方案、时间点、分析方法是否有选择性。若有预先计划且报告一致，倾向 Low；若方案或注册信息不足，倾向 Some concerns；若明显选择性报告，倾向 High。
""".strip()

EVALUATOR_SYSTEM_PROMPT = """
你现在是一位非常严苛、注重细节、专门负责挑刺的 Cochrane 循证医学方法学家。你的任务不是给作者好评，而是主动寻找偏倚风险。你必须仔细阅读我提供的随机对照试验（RCT）全文，并严格按照 Cochrane Risk of Bias 2 (RoB 2) 工具的标准，对 5 个领域及总体偏倚进行风险评估。

【RoB 2 评价标准指引】：
- D1 随机化过程的偏倚：重点寻找分配序列如何产生、分配隐藏方法，以及基线特征是否平衡。
- D2 偏离既定干预措施的偏倚：重点寻找受试者和实施者是否盲法，是否发生方案偏离，以及是否采用意向性分析（ITT）评估分配干预的效果。
- D3 结局数据缺失的偏倚：重点寻找受试者脱落率、两组缺失比例是否平衡、缺失原因，以及是否进行了合理的插补分析。
- D4 结局测量的偏倚：重点寻找结局评估者是否盲法，测量方法是否客观并在两组间保持一致。
- D5 报告结果选择的偏倚：重点寻找是否有预先注册的试验方案，报告的结局和分析方法是否被选择性挑出（如只报告阳性结果）。

【总体偏倚 (Overall) 判定严格逻辑】：
- Low：所有 D1-D5 均为 Low。
- Some concerns：至少 1 个维度是 Some concerns，且没有任何维度是 High。
- High：至少 1 个维度是 High，或多个维度是 Some concerns 从而实质性降低了整体信心。

【强制执行路径】：
""" + "\n" + ROB2_STRICT_WORKFLOW + "\n\n" + """
【逐域量化尺子】：
""" + "\n" + ROB2_DOMAIN_RUBRIC + "\n\n" + """

补充要求：
- 你必须结合全文证据独立判断，不要迎合其他评价员，也不要为了省事跳过任一维度。
- 你必须在内部 step by step 完成逐域分析，再输出最终 JSON；不要把内部推理过程输出到 JSON 之外。
- 优先引用或紧贴原文证据；若文中未报告，请直接写“未报告”或“信息不足”。
- 每个 `reason` 必须控制在 50 个汉字以内。
- 所有键都必须完整输出，缺失任意一个键都视为失败。
- 不允许空字符串、`null`、`N/A`、`unknown` 之类占位值。
- 每个 `score` 的值必须且只能是 "Low"、"Some concerns" 或 "High"。
- 不要输出 JSON 以外的任何文字。

请严格输出以下 JSON：
{
  "D1_score": "Low/Some concerns/High", "D1_reason": "不超过50个字的理由",
  "D2_score": "Low/Some concerns/High", "D2_reason": "不超过50个字的理由",
  "D3_score": "Low/Some concerns/High", "D3_reason": "不超过50个字的理由",
  "D4_score": "Low/Some concerns/High", "D4_reason": "不超过50个字的理由",
  "D5_score": "Low/Some concerns/High", "D5_reason": "不超过50个字的理由",
  "Overall_score": "Low/Some concerns/High", "Overall_reason": "不超过50个字的理由"
}
""".strip()

AGGREGATOR_SYSTEM_PROMPT = """
你现在是一位拥有 20 年经验、非常严苛且注重证据优先的 Cochrane 主审专家。你将接收到 3 位独立评价员对同一篇 RCT 文献的 RoB 2 独立打分及提取的理由。你的任务是仲裁，而不是复述。

你的任务是：
1. 审阅这 3 份 JSON 打分表。
2. 对 D1-D5 每个维度分别比较 3 位评价员的分歧，遵循“少数服从多数”与“证据优先”原则完成裁决。
3. 严格遵循 RoB 2 总体偏倚计算逻辑：只要有一个维度裁定为 High，总体必为 High；只要所有维度均为 Low，总体才为 Low。

请同时参考以下 RoB 2 原则进行仲裁：
""" + "\n" + ROB2_GUIDANCE_SUMMARY + "\n\n" + """
【强制执行路径】：
- 先逐个维度比较 3 份评分与理由，再输出最终 D1-D5。
- 如果多数意见与最直接原文证据冲突，优先采纳证据更充分的一方。
- 完成 D1-D5 后，最后一步才计算 Overall。
- 你必须在内部 step by step 完成比较和裁决，但不要输出内部推理。

输出要求：
- 严格输出一个纯净 JSON 对象。
- 不要包含解释文字、前后缀或 Markdown。
- 每个键的值必须且只能是 "Low"、"Some concerns" 或 "High"。

请输出：
{
  "D1": "Low/Some concerns/High",
  "D2": "Low/Some concerns/High",
  "D3": "Low/Some concerns/High",
  "D4": "Low/Some concerns/High",
  "D5": "Low/Some concerns/High",
  "Overall": "Low/Some concerns/High"
}
""".strip()


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model_name: str


def is_retryable_exception(exc: Exception) -> bool:
    """仅对限流、超时、连接失败、服务端错误和 JSON 解析问题重试。"""
    if isinstance(
        exc,
        (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            json.JSONDecodeError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def ensure_dependencies() -> bool:
    """检查关键依赖是否可用。"""
    missing = []
    if fitz is None:
        missing.append("pymupdf")
    if pdfplumber is None:
        missing.append("pdfplumber")
    if OpenAI is None:
        missing.append("openai")
    if retry is None:
        missing.append("tenacity")

    if missing:
        st.error(
            "缺少运行依赖："
            + ", ".join(missing)
            + "。请先安装后再运行，例如：`pip install streamlit pandas openai tenacity pymupdf pdfplumber pypdf`"
        )
        return False
    return True


def build_error_result(message: str = "API请求失败或超时") -> dict[str, Any]:
    return {
        "score": 0,
        "relevance": "None",
        "decision": "Error",
        "rationale": "API请求失败或超时",
        "error": message[:200] if message else "API请求失败或超时",
    }


def short_error_message(exc: Exception) -> str:
    """提取更适合在界面中展示的错误信息。"""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message") or body.get("error") or body
        if isinstance(message, dict):
            message = message.get("message") or message
        if message:
            return str(message)

    message = getattr(exc, "message", None)
    if message:
        return str(message)

    text = str(exc).strip()
    return text or exc.__class__.__name__


def load_csv_file(uploaded_file: Any) -> pd.DataFrame:
    """尝试用常见编码读取 CSV，提升 Zotero/EndNote 导出兼容性。"""
    file_bytes = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            text = file_bytes.decode(encoding)
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ValueError(f"无法读取 CSV 文件，请检查编码或文件格式。原始错误：{last_error}")


def decode_text_bytes(file_bytes: bytes) -> str:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ValueError(f"无法读取全文文本文件，请检查编码。原始错误：{last_error}")


def normalize_study_name(raw_name: str, fallback: str) -> str:
    study = str(raw_name or "").strip() or fallback
    study = re.sub(r"^(mineru[_ -]*markdown[_ -]*|mineru[_ -]*|markdown[_ -]*)+", "", study, flags=re.I)
    study = re.sub(r"_\d{10,}$", "", study)
    study = re.sub(r"[_]+", " ", study)
    study = re.sub(r"\s+", " ", study).strip(" -_")
    return study or fallback


def parse_local_fulltext_paths(raw_value: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    for line in str(raw_value or "").splitlines():
        cleaned = line.strip().strip('"').strip("'")
        if not cleaned:
            continue
        path = Path(cleaned).expanduser()
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        paths.append(path)

    return paths


def normalize_extracted_fulltext_text(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", cleaned)
    cleaned = cleaned.replace("```", "\n")
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*[-*_]{3,}\s*$", "", cleaned, flags=re.M)

    noise_patterns = [
        r"^To cite this article:.*$",
        r"^To link to this article:.*$",
        r"^Submit your article to this journal.*$",
        r"^View related articles.*$",
        r"^View Crossmark data.*$",
        r"^Article views:.*$",
        r"^Published online:.*$",
        r"^\s*[•◦]\s*$",
    ]
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.M)

    lines: list[str] = []
    previous_line = ""
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell and not re.fullmatch(r"[:\- ]+", cell)]
            line = "\t".join(cells)
        else:
            line = re.sub(r"\s+", " ", line)

        if not line or line == previous_line:
            continue

        previous_line = line
        lines.append(line)

    return "\n".join(lines).strip()


def extract_text_from_preparsed_file(file_bytes: bytes, suffix: str) -> tuple[str, int]:
    raw_text = decode_text_bytes(file_bytes)
    normalized_text = normalize_extracted_fulltext_text(raw_text)
    if not normalized_text:
        raise ValueError(f"{suffix} 文件未提取到可用全文内容。")
    return normalized_text, 0


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df.columns = [str(col).replace("\ufeff", "").strip() for col in cleaned_df.columns]
    return cleaned_df


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_choice(value: Any, aliases: dict[str, str], valid_values: set[str], field_name: str) -> str:
    text = str(value).strip()
    if text in valid_values:
        return text

    normalized = aliases.get(text.lower())
    if normalized:
        return normalized

    raise ValueError(f"{field_name} 不在允许范围内。")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验必需列，并过滤掉标题和摘要都为空的记录。"""
    cleaned_df = sanitize_columns(df)
    required_columns = {"Title", "Abstract Note"}
    missing_columns = required_columns.difference(cleaned_df.columns)
    if missing_columns:
        missing_text = "、".join(sorted(missing_columns))
        raise ValueError(
            f"CSV 缺少必需列：{missing_text}。请确认导出文件中包含 `Title` 和 `Abstract Note` 两列。"
        )

    working_df = cleaned_df.copy()
    working_df["Title"] = working_df["Title"].apply(normalize_text)
    working_df["Abstract Note"] = working_df["Abstract Note"].apply(normalize_text)

    valid_mask = ~((working_df["Title"] == "") & (working_df["Abstract Note"] == ""))
    filtered_df = working_df.loc[valid_mask].reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError("有效文献数为 0。所有记录的 `Title` 与 `Abstract Note` 都为空。")

    return filtered_df


def build_screening_messages(topic: str, title: str, abstract: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是一位严谨的临床研究者。你的任务是评估文献与以下【核心课题】的匹配程度：\n"
        f"{topic}\n\n"
        "请进行 0-100 分的综合打分：\n"
        "- 90-100分 (Include)：明确研究了该领域，且重点探讨了相关的标志物、机制或精准干预。\n"
        "- 70-89分 (Include)：高度相关或有潜在价值的间接证据。\n"
        "- 40-69分 (Unsure)：涉及相关疾病领域但未提及核心要素，需看全文。\n"
        "- 0-39分 (Exclude)：完全无关，或属于动物/体外实验、无数据综述等排除项。\n\n"
        "必须以严格的 JSON 格式输出，包含以下四个键值：\n"
        "'score': 分数(整数),\n"
        "'relevance': 'High'/'Medium'/'Low'/'None',\n"
        "'decision': 'Include'/'Unsure'/'Exclude',\n"
        "'rationale': 判定理由(50字内，明确指出是否包含标志物或触及排除标准)。"
    )
    user_prompt = f"标题：{title}\n摘要：{abstract}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def preview_text(text: str, limit: int = 1200) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...[truncated]"


def truncate_text(text: str, max_chars: int) -> str:
    compact = str(text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n\n[内容已截断以保证格式化重试稳定性]"


def canonicalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def flatten_json_pairs(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(value, dict):
        return []

    items: list[tuple[str, Any]] = []
    for key, item in value.items():
        full_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            items.extend(flatten_json_pairs(item, full_key))
        else:
            items.append((full_key, item))
    return items


def lookup_case_insensitive(mapping: dict[str, Any], target: str) -> Any:
    target_key = canonicalize_key(target)
    for key, value in mapping.items():
        if canonicalize_key(key) == target_key:
            return value
    return None


def extract_domain_value(
    result: dict[str, Any],
    domain: str,
    value_type: str,
    flat_lookup: dict[str, Any],
) -> Any:
    domain_payload = lookup_case_insensitive(result, domain)
    if isinstance(domain_payload, dict):
        nested_value = (
            lookup_case_insensitive(domain_payload, value_type)
            or lookup_case_insensitive(domain_payload, f"{domain}_{value_type}")
        )
        if nested_value is not None:
            return nested_value

    containers = {
        "score": ["scores", "ratings", "judgements", "judgments", "assessment", "assessments"],
        "reason": ["reasons", "rationales", "evidence", "justifications", "explanations"],
    }
    for container_name in containers[value_type]:
        container = lookup_case_insensitive(result, container_name)
        if isinstance(container, dict):
            nested_value = lookup_case_insensitive(container, domain)
            if nested_value is not None:
                return nested_value

    direct_candidates = [
        f"{domain}_{value_type}",
        f"{value_type}_{domain}",
        f"{domain}{value_type}",
        f"{value_type}{domain}",
    ]
    if value_type == "reason":
        direct_candidates.extend(
            [
                f"{domain}_rationale",
                f"{domain}rationale",
                f"rationale_{domain}",
                f"evidence_{domain}",
                f"{domain}_evidence",
            ]
        )

    for candidate in direct_candidates:
        value = flat_lookup.get(canonicalize_key(candidate))
        if value is not None:
            return value

    if value_type == "score" and isinstance(domain_payload, str):
        return domain_payload

    return None


def split_text_into_overlapping_chunks(
    text: str,
    chunk_size: int = ROB_EVIDENCE_CHUNK_SIZE,
    overlap: int = ROB_EVIDENCE_CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start:start + chunk_size]
        if len(chunk) < 400 and start > 0:
            break
        chunks.append((start, chunk))
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def score_rob_evidence_chunk(chunk: str, start: int) -> float:
    lowered = chunk.lower()
    score = 0.0
    for keyword, weight in ROB_EVIDENCE_KEYWORDS.items():
        hits = lowered.count(keyword)
        score += min(hits, 4) * weight

    if start == 0:
        score += 2.0
    if re.search(r"\bn\s*=\s*\d+", lowered):
        score += 1.0
    if re.search(r"\b\d{1,3}(?:\.\d+)?%", chunk):
        score += 1.0
    if "references" in lowered and start > len(chunk) * 2:
        score -= 2.0
    return score


def build_rob_evidence_pack(pdf_text: str) -> str:
    chunks = split_text_into_overlapping_chunks(pdf_text)
    if not chunks:
        return pdf_text

    if len(pdf_text) <= ROB_EVIDENCE_MAX_CHARS:
        return pdf_text

    ranked = sorted(
        ((score_rob_evidence_chunk(chunk, start), start, chunk) for start, chunk in chunks),
        key=lambda item: (-item[0], item[1]),
    )

    selected_starts = {0}
    total_chars = len(chunks[0][1])
    for score, start, chunk in ranked:
        if score <= 0 or start in selected_starts:
            continue
        if total_chars + len(chunk) > ROB_EVIDENCE_MAX_CHARS:
            continue
        selected_starts.add(start)
        total_chars += len(chunk)
        if total_chars >= ROB_EVIDENCE_MAX_CHARS:
            break

    selected_chunks = [chunk for start, chunk in chunks if start in selected_starts]
    header = (
        "以下内容是从全文中提取并按 RoB 2 相关性优先保留的证据摘录，"
        "重点覆盖随机化、盲法、偏离方案、缺失数据、结局测量、分析计划与注册信息：\n\n"
    )
    return header + "\n\n".join(selected_chunks)


def normalize_pdf_body_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
    cleaned = re.sub(r"[\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff.%<>/=+-]", " ", cleaned)
    cleaned = re.sub(r"_+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_pdf_table_cell(value: Any) -> str:
    cleaned = str(value or "")
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff.%<>/=+-]", " ", cleaned)
    cleaned = re.sub(r"_+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_pdf_text_with_pdfplumber(file_bytes: bytes) -> tuple[str, int]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber 依赖未安装。")

    body_segments: list[str] = []
    table_segments: list[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            page_text = " ".join(
                cleaned for word in words if (cleaned := normalize_pdf_body_text(word.get("text", "")))
            )
            if page_text:
                body_segments.append(page_text)

            try:
                tables = page.extract_tables()
            except Exception:  # noqa: BLE001
                tables = []

            for table_index, table in enumerate(tables, start=1):
                rows = []
                for row in table or []:
                    cells = [normalize_pdf_table_cell(cell) for cell in (row or [])]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        rows.append("\t".join(cells))
                if rows:
                    table_segments.append(
                        f"Table Page {page_index} Index {table_index}\n" + "\n".join(rows)
                    )

    combined_parts = []
    if body_segments:
        combined_parts.append("Body Content\n" + "\n\n".join(body_segments))
    if table_segments:
        combined_parts.append("Table Data\n" + "\n\n".join(table_segments))

    combined_text = "\n\n".join(part for part in combined_parts if part).strip()
    if not combined_text:
        raise ValueError("PDF 未提取到正文或表格数据。")
    return combined_text, page_count


def split_pdf_cleanup_chunks(text: str, max_chars: int = PDF_LLM_EXTRACTION_CHUNK_CHARS) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", str(text or "")) if block.strip()]
    if not blocks:
        return []

    chunks: list[str] = []
    current_blocks: list[str] = []
    current_length = 0

    for block in blocks:
        if len(block) > max_chars:
            if current_blocks:
                chunks.append("\n\n".join(current_blocks))
                current_blocks = []
                current_length = 0
            for start in range(0, len(block), max_chars):
                part = block[start:start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        added_length = len(block) + (2 if current_blocks else 0)
        if current_blocks and current_length + added_length > max_chars:
            chunks.append("\n\n".join(current_blocks))
            current_blocks = [block]
            current_length = len(block)
            continue

        current_blocks.append(block)
        current_length += added_length

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks


def finalize_llm_pdf_text(text: str) -> str:
    lines: list[str] = []
    previous_line = ""

    for raw_line in str(text or "").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        if "\t" in stripped_line:
            cells = [normalize_pdf_table_cell(cell) for cell in stripped_line.split("\t")]
            cells = [cell for cell in cells if cell]
            normalized_line = "\t".join(cells)
        else:
            normalized_line = normalize_pdf_body_text(stripped_line)

        if not normalized_line or normalized_line == previous_line:
            continue

        previous_line = normalized_line
        lines.append(normalized_line)

    return "\n".join(lines).strip()


def prepare_llm_pdf_extraction_input(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return text

    if len(text) <= PDF_LLM_EXTRACTION_SOURCE_MAX_CHARS:
        return text

    return build_rob_evidence_pack(text)


def normalize_pdf_extraction_result(result: dict[str, Any]) -> dict[str, str]:
    if not isinstance(result, dict):
        raise ValueError("PDF 提取增强返回结果不是 JSON 对象。")

    clean_text = (
        lookup_case_insensitive(result, "clean_text")
        or lookup_case_insensitive(result, "text")
        or lookup_case_insensitive(result, "content")
    )
    if clean_text is None:
        raise ValueError("PDF 提取增强返回 JSON 缺少字段：clean_text。")

    normalized_text = finalize_llm_pdf_text(str(clean_text))
    if not normalized_text:
        raise ValueError("PDF 提取增强 clean_text 为空。")

    return {"clean_text": normalized_text}


def build_pdf_extraction_messages(
    study: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> list[dict[str, str]]:
    user_prompt = (
        f"研究名称：{study}\n"
        f"当前处理块：{chunk_index}/{total_chunks}\n"
        "以下是从专业论文 PDF 中解析出的原始正文与表格文本。"
        "请只重建正文内容和专业表格数据，删除页眉页脚、重复碎片和普通标点噪声，"
        "但保留专业数值、小数、百分比、单位和表格结构。\n\n"
        f"{chunk_text}"
    )
    return [
        {"role": "system", "content": PDF_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def llm_refine_pdf_text(
    config: ApiConfig,
    study: str,
    raw_text: str,
    progress_queue: queue.Queue | None = None,
) -> str:
    llm_source_text = prepare_llm_pdf_extraction_input(raw_text)
    chunks = split_pdf_cleanup_chunks(llm_source_text)
    if not chunks:
        raise ValueError("PDF 原始提取文本为空，无法执行大模型增强。")

    refined_chunks: list[str] = []
    total_chunks = len(chunks)
    LOGGER.info(
        "PDF LLM refine start | study=%s | chunks=%s | raw_chars=%s | llm_source_chars=%s",
        study,
        total_chunks,
        len(raw_text),
        len(llm_source_text),
    )

    for index, chunk in enumerate(chunks, start=1):
        if progress_queue is not None:
            progress_queue.put(
                {
                    "study": study,
                    "stage": "extracting_llm",
                    "chunk": index,
                    "total": total_chunks,
                }
            )

        try:
            result = call_json_with_retry(
                config=config,
                messages=build_pdf_extraction_messages(study, chunk, index, total_chunks),
                normalizer=normalize_pdf_extraction_result,
                temperature=0.0,
                timeout_seconds=PDF_LLM_EXTRACTION_TIMEOUT_SECONDS,
                max_tokens=PDF_LLM_EXTRACTION_MAX_TOKENS,
            )
            refined_chunks.append(result["clean_text"])
            LOGGER.info(
                "PDF LLM refine success | study=%s | chunk=%s/%s | chars=%s",
                study,
                index,
                total_chunks,
                len(result["clean_text"]),
            )
        except Exception as exc:  # noqa: BLE001
            fallback_chunk = finalize_llm_pdf_text(chunk)
            if fallback_chunk:
                refined_chunks.append(fallback_chunk)
            LOGGER.warning(
                "PDF LLM refine failed, fallback to raw chunk | study=%s | chunk=%s/%s | error=%s",
                study,
                index,
                total_chunks,
                short_error_message(exc),
            )

    final_text = finalize_llm_pdf_text("\n\n".join(chunk for chunk in refined_chunks if chunk.strip()))
    if not final_text:
        raise ValueError("大模型辅助 PDF 提取后为空。")
    LOGGER.info("PDF LLM refine completed | study=%s | chars=%s", study, len(final_text))
    return final_text


def normalize_screening_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("模型返回结果不是 JSON 对象。")

    if "score" not in result or "relevance" not in result or "decision" not in result or "rationale" not in result:
        raise ValueError("模型返回 JSON 缺少必要字段。")

    try:
        score = int(result["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("score 必须是整数。") from exc

    score = max(0, min(100, score))
    relevance = normalize_choice(result["relevance"], RELEVANCE_ALIASES, VALID_RELEVANCE, "relevance")
    decision = normalize_choice(
        result["decision"],
        DECISION_ALIASES,
        VALID_DECISION - {"Error"},
        "decision",
    )
    rationale = str(result["rationale"]).strip()

    if not rationale:
        raise ValueError("rationale 不能为空。")

    return {
        "score": score,
        "relevance": relevance,
        "decision": decision,
        "rationale": rationale[:50],
        "error": "",
    }


def normalize_rob_choice(value: Any, field_name: str) -> str:
    text = str(value).strip()
    text = text.strip('",. \t\n')

    if text in ROB_ALLOWED_SCORES:
        return text

    normalized = ROB_SCORE_ALIASES.get(text.lower())
    if normalized:
        return normalized

    fallback_score = "Some concerns"
    LOGGER.warning(
        "硬容错触发：检测到非法评分值 '%s' (%s)，已自动降级为 '%s'",
        value,
        field_name,
        fallback_score,
    )
    return fallback_score


def derive_overall_rob_score(domain_scores: dict[str, str]) -> str:
    evaluator_scores = [domain_scores.get(domain, "Some concerns") for domain in ROB_EVALUATOR_DOMAINS]
    if all(score == "Low" for score in evaluator_scores):
        return "Low"
    if any(score == "High" for score in evaluator_scores):
        return "High"
    if sum(score == "Some concerns" for score in evaluator_scores) >= 2:
        return "High"
    return "Some concerns"


def fallback_rob_reason(field_name: str, score: str = "Some concerns") -> str:
    if field_name == "Overall_reason":
        if score == "Low":
            return "各域均低风险"
        if score == "High":
            return "至少一域高风险或多域存疑"
        return "至少一域存疑"
    return "信息不足"


def choose_consensus_rob_score(scores: list[str], domain: str) -> str:
    if not scores:
        LOGGER.warning("Aggregator 共识补齐时缺少 %s 候选评分，默认 Some concerns", domain)
        return "Some concerns"

    rank = {"Low": 0, "Some concerns": 1, "High": 2}
    counts = Counter(scores)
    max_count = max(counts.values())
    candidates = [score for score, count in counts.items() if count == max_count]
    chosen = max(candidates, key=lambda score: rank.get(score, 1))
    LOGGER.warning(
        "Aggregator 缺少字段：%s，已按 3 位评价员多数票补齐为 %s | candidates=%s",
        domain,
        chosen,
        scores,
    )
    return chosen


def build_aggregator_fallback_result(evaluator_results: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: dict[str, str] = {}
    for domain in ROB_EVALUATOR_DOMAINS:
        scores = [
            normalize_rob_choice(result.get(f"{domain}_score", "Some concerns"), f"{domain}_score")
            for result in evaluator_results
            if isinstance(result, dict)
        ]
        normalized[domain] = choose_consensus_rob_score(scores, domain)

    normalized["Overall"] = derive_overall_rob_score(normalized)
    return normalized


def normalize_evaluator_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Evaluator 返回结果不是 JSON 对象。")

    flat_lookup = {canonicalize_key(key): value for key, value in flatten_json_pairs(result)}
    normalized: dict[str, Any] = {}
    domain_scores: dict[str, str] = {}

    for domain in ROB_EVALUATOR_DOMAINS:
        score_key = f"{domain}_score"
        reason_key = f"{domain}_reason"
        score_value = extract_domain_value(result, domain, "score", flat_lookup)
        reason_value = extract_domain_value(result, domain, "reason", flat_lookup)

        if score_value is None:
            LOGGER.warning("Evaluator 缺少字段：%s，已自动填充为 Some concerns", score_key)
            normalized_score = "Some concerns"
        else:
            normalized_score = normalize_rob_choice(score_value, score_key)

        reason = str(reason_value).strip()[:50] if reason_value is not None else ""
        if not reason:
            reason = fallback_rob_reason(reason_key, normalized_score)
            LOGGER.warning("Evaluator 缺少字段：%s，已自动填充为 %s", reason_key, reason)

        normalized[score_key] = normalized_score
        normalized[reason_key] = reason
        domain_scores[domain] = normalized_score

    overall_score_key = "Overall_score"
    overall_reason_key = "Overall_reason"
    overall_score_value = extract_domain_value(result, "Overall", "score", flat_lookup)
    overall_reason_value = extract_domain_value(result, "Overall", "reason", flat_lookup)
    derived_overall_score = derive_overall_rob_score(domain_scores)

    if overall_score_value is None:
        LOGGER.warning("Evaluator 缺少字段：%s，已按 D1-D5 自动计算为 %s", overall_score_key, derived_overall_score)
        overall_score = derived_overall_score
    else:
        overall_score = normalize_rob_choice(overall_score_value, overall_score_key)
        if overall_score != derived_overall_score:
            LOGGER.warning(
                "Evaluator Overall 与 D1-D5 不一致：原值=%s，已按规则改写为 %s",
                overall_score,
                derived_overall_score,
            )
            overall_score = derived_overall_score

    overall_reason = str(overall_reason_value).strip()[:50] if overall_reason_value is not None else ""
    if not overall_reason:
        overall_reason = fallback_rob_reason(overall_reason_key, overall_score)
        LOGGER.warning("Evaluator 缺少字段：%s，已自动填充为 %s", overall_reason_key, overall_reason)

    normalized[overall_score_key] = overall_score
    normalized[overall_reason_key] = overall_reason

    return normalized


def normalize_aggregator_result(
    result: dict[str, Any],
    evaluator_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Aggregator 返回结果不是 JSON 对象。")

    flat_lookup = {canonicalize_key(key): value for key, value in flatten_json_pairs(result)}
    normalized: dict[str, Any] = {}
    fallback_from_evaluators = (
        build_aggregator_fallback_result(evaluator_results)
        if evaluator_results
        else {domain: "Some concerns" for domain in ROB_EVALUATOR_DOMAINS}
    )
    for domain in ROB_EVALUATOR_DOMAINS:
        value = extract_domain_value(result, domain, "score", flat_lookup)
        if value is None:
            value = lookup_case_insensitive(result, domain)
        if value is None:
            normalized[domain] = fallback_from_evaluators[domain]
            LOGGER.warning("Aggregator 缺少字段：%s，已自动补齐为 %s", domain, normalized[domain])
            continue
        normalized[domain] = normalize_rob_choice(value, domain)

    overall_value = extract_domain_value(result, "Overall", "score", flat_lookup)
    if overall_value is None:
        overall_value = lookup_case_insensitive(result, "Overall")

    derived_overall_score = derive_overall_rob_score(normalized)
    if overall_value is None:
        LOGGER.warning("Aggregator 缺少字段：Overall，已按 D1-D5 自动计算为 %s", derived_overall_score)
        normalized["Overall"] = derived_overall_score
    else:
        overall_score = normalize_rob_choice(overall_value, "Overall")
        if overall_score != derived_overall_score:
            LOGGER.warning(
                "Aggregator Overall 与 D1-D5 不一致：原值=%s，已按规则改写为 %s",
                overall_score,
                derived_overall_score,
            )
            overall_score = derived_overall_score
        normalized["Overall"] = overall_score

    return normalized


def extract_message_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()

    if isinstance(message_content, dict):
        if isinstance(message_content.get("text"), str):
            return message_content["text"]
        if isinstance(message_content.get("content"), str):
            return message_content["content"]

    return json.dumps(message_content, ensure_ascii=False)


def _call_json_once(
    config: ApiConfig,
    messages: list[dict[str, str]],
    normalizer: JsonNormalizer,
    temperature: float = 0.0,
    timeout_seconds: float = SCREENING_TIMEOUT_SECONDS,
    max_tokens: int = SCREENING_MAX_TOKENS,
) -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai 依赖未安装。")

    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout_seconds)
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)

    content = extract_message_content(response.choices[0].message.content) or "{}"
    json_text = extract_json_text(content)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        LOGGER.exception("JSON decode failed | preview=%s", preview_text(content))
        raise

    try:
        return normalizer(parsed)
    except Exception:
        LOGGER.exception(
            "JSON normalize failed | top_keys=%s | preview=%s",
            list(parsed.keys())[:20] if isinstance(parsed, dict) else type(parsed).__name__,
            preview_text(content),
        )
        raise


if retry is not None:

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def call_json_with_retry(
        config: ApiConfig,
        messages: list[dict[str, str]],
        normalizer: JsonNormalizer,
        temperature: float = 0.0,
        timeout_seconds: float = SCREENING_TIMEOUT_SECONDS,
        max_tokens: int = SCREENING_MAX_TOKENS,
    ) -> dict[str, Any]:
        return _call_json_once(config, messages, normalizer, temperature, timeout_seconds, max_tokens)

else:

    def call_json_with_retry(
        config: ApiConfig,
        messages: list[dict[str, str]],
        normalizer: JsonNormalizer,
        temperature: float = 0.0,
        timeout_seconds: float = SCREENING_TIMEOUT_SECONDS,
        max_tokens: int = SCREENING_MAX_TOKENS,
    ) -> dict[str, Any]:
        return _call_json_once(config, messages, normalizer, temperature, timeout_seconds, max_tokens)


def screen_single_paper(config: ApiConfig, topic: str, title: str, abstract: str) -> dict[str, Any]:
    """单篇文献筛选，失败时返回默认结果，避免整批任务中断。"""
    try:
        return call_json_with_retry(
            config=config,
            messages=build_screening_messages(topic, title, abstract),
            normalizer=normalize_screening_result,
            temperature=0.0,
            timeout_seconds=SCREENING_TIMEOUT_SECONDS,
            max_tokens=SCREENING_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        return build_error_result(short_error_message(exc))


def test_api_connection(config: ApiConfig, topic: str) -> str | None:
    """批量处理前先做一次轻量 API 预检，便于提前暴露配置问题。"""
    try:
        call_json_with_retry(
            config=config,
            messages=build_screening_messages(
                topic=topic,
                title="连接测试：基于标志物重新定义肥胖的临床研究",
                abstract="本条仅用于验证 API 参数、模型名称与 JSON 输出能力是否正常。",
            ),
            normalizer=normalize_screening_result,
            temperature=0.0,
            timeout_seconds=SCREENING_TIMEOUT_SECONDS,
            max_tokens=SCREENING_MAX_TOKENS,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        return short_error_message(exc)


def run_batch_screening(
    df: pd.DataFrame,
    config: ApiConfig,
    topic: str,
    max_workers: int,
    progress_bar: Any,
    status_placeholder: Any,
) -> pd.DataFrame:
    records = df[["Title", "Abstract Note"]].to_dict("records")
    total = len(records)
    results = [build_error_result() for _ in range(total)]

    if total == 0:
        return df.copy()

    progress_bar.progress(0)
    status_placeholder.info(f"正在处理: 0 / {total} 篇")

    worker_count = min(max_workers, total)
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                screen_single_paper,
                config,
                topic,
                record["Title"],
                record["Abstract Note"],
            ): idx
            for idx, record in enumerate(records)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception:  # noqa: BLE001
                results[index] = build_error_result()

            completed += 1
            progress_bar.progress(completed / total)
            status_placeholder.info(f"正在处理: {completed} / {total} 篇")

    result_df = df.copy()
    result_df["AI Score"] = [item["score"] for item in results]
    result_df["AI Relevance"] = [item["relevance"] for item in results]
    result_df["AI Decision"] = [item["decision"] for item in results]
    result_df["AI Rationale"] = [item["rationale"] for item in results]
    result_df["AI Error"] = [item.get("error", "") for item in results]
    return result_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def extract_pdf_text(
    file_bytes: bytes,
    *,
    config: ApiConfig | None = None,
    study: str = "",
    use_llm_refinement: bool = False,
    progress_queue: queue.Queue | None = None,
) -> tuple[str, int]:
    extraction_errors: list[str] = []
    raw_text: str | None = None
    page_count = 0

    if pdfplumber is not None:
        try:
            raw_text, page_count = extract_pdf_text_with_pdfplumber(file_bytes)
        except Exception as exc:  # noqa: BLE001
            extraction_errors.append(f"pdfplumber: {short_error_message(exc)}")
            LOGGER.warning("pdfplumber 提取失败，回退到 fitz | error=%s", short_error_message(exc))

    if not raw_text and fitz is None:
        joined_errors = " | ".join(extraction_errors) if extraction_errors else "未安装 pymupdf"
        raise RuntimeError(f"PDF 提取器不可用：{joined_errors}")

    if not raw_text:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            page_count = document.page_count
            pages = [normalize_pdf_body_text(page.get_text("text")) for page in document]
        finally:
            document.close()
        raw_text = "\n".join(part for part in pages if part).strip()

    if not raw_text:
        joined_errors = " | ".join(extraction_errors) if extraction_errors else "无"
        raise ValueError(f"PDF 未提取到可用正文或表格数据。前序错误：{joined_errors}")

    if use_llm_refinement and config is not None:
        if progress_queue is not None:
            progress_queue.put({"study": study, "stage": "extracting_llm", "chunk": 0, "total": 0})
        try:
            refined_text = llm_refine_pdf_text(config, study or "Unknown Study", raw_text, progress_queue)
            return refined_text, page_count
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("大模型辅助 PDF 提取失败，回退到原始解析文本 | study=%s | error=%s", study, short_error_message(exc))

    return raw_text, page_count


def build_evaluator_messages(
    study: str,
    pdf_text: str,
    reviewer_id: int,
    *,
    format_retry: bool = False,
    last_error: str = "",
) -> list[dict[str, str]]:
    retry_note = ""
    if format_retry:
        retry_note = (
            "\n\n这是一次字段补救重试。你上一次输出因 JSON 字段缺失而无效。"
            "这一次必须完整输出全部 12 个键。"
            "如果证据不足，reason 填“信息不足”；如果难以确定评分，score 优先填“Some concerns”，但绝不允许留空或省略键。"
        )
        if last_error:
            retry_note += f"\n上一次错误：{last_error[:120]}"

    user_prompt = (
        f"研究名称：{study}\n"
        f"当前身份：独立初级评价员 #{reviewer_id}\n"
        f"证据长度：约 {len(pdf_text):,} 字符\n"
        "以下是通过 PDF 全文提取并按 RoB 2 相关性优先保留的证据摘录。"
        "请基于这些全文证据完成 D1-D5 与 Overall 的完整 JSON 评分。"
        "你必须逐个维度完成：定位证据 -> 判断风险 -> 压缩理由 -> 最后计算 Overall。"
        "如果某一维度证据不足，请在对应 reason 中明确写“未报告”或“信息不足”，不要自行脑补。"
        "提交前请自检：12 个键全部输出、没有空值、没有 null、score 只用 Low/Some concerns/High，且每个 reason 不超过 50 个汉字。"
        f"{retry_note}\n\n"
        f"{pdf_text}"
    )
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_aggregator_messages(
    study: str,
    evaluator_results: list[dict[str, Any]],
    *,
    format_retry: bool = False,
    last_error: str = "",
) -> list[dict[str, str]]:
    reviewer_blocks = []
    for index, result in enumerate(evaluator_results, start=1):
        reviewer_blocks.append(
            f"Evaluator_{index} JSON:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
        )

    retry_note = ""
    if format_retry:
        retry_note = (
            "\n\n这是一次字段补救重试。你上一次输出缺少必要字段。"
            "这一次必须只输出 D1、D2、D3、D4、D5、Overall 这 6 个键，且每个值只能是 Low / Some concerns / High。"
        )
        if last_error:
            retry_note += f"\n上一次错误：{last_error[:120]}"

    user_prompt = (
        f"研究名称：{study}\n"
        "以下是 3 位独立评价员对同一篇 RCT 的 RoB 2 JSON 打分表。"
        "请严格依据它们的评分与理由，逐个维度完成仲裁，并输出 D1-D5 与 Overall 的最终评级。"
        "提交前请自检：只输出 6 个键，且每个值只能是 Low/Some concerns/High。"
        f"{retry_note}\n\n"
        + "\n\n".join(reviewer_blocks)
    )
    return [
        {"role": "system", "content": AGGREGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def evaluate_single_reviewer(
    config: ApiConfig,
    study: str,
    pdf_text: str,
    reviewer_id: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    LOGGER.info(
        "Evaluator start | study=%s | reviewer=%s | chars=%s",
        study,
        reviewer_id,
        len(pdf_text),
    )
    try:
        result = call_json_with_retry(
            config=config,
            messages=build_evaluator_messages(study, pdf_text, reviewer_id),
            normalizer=normalize_evaluator_result,
            temperature=EVALUATOR_TEMPERATURE,
            timeout_seconds=EVALUATOR_TIMEOUT_SECONDS,
            max_tokens=EVALUATOR_MAX_TOKENS,
        )
        LOGGER.info(
            "Evaluator success | study=%s | reviewer=%s | elapsed=%.2fs",
            study,
            reviewer_id,
            time.perf_counter() - started_at,
        )
        return result
    except Exception as exc:
        LOGGER.warning(
            "Evaluator format fallback | study=%s | reviewer=%s | error=%s",
            study,
            reviewer_id,
            short_error_message(exc),
        )
        retry_text = truncate_text(pdf_text, FORMAT_RETRY_EVIDENCE_MAX_CHARS)
        try:
            result = call_json_with_retry(
                config=config,
                messages=build_evaluator_messages(
                    study,
                    retry_text,
                    reviewer_id,
                    format_retry=True,
                    last_error=short_error_message(exc),
                ),
                normalizer=normalize_evaluator_result,
                temperature=EVALUATOR_TEMPERATURE,
                timeout_seconds=FORMAT_RETRY_TIMEOUT_SECONDS,
                max_tokens=FORMAT_RETRY_MAX_TOKENS,
            )
            LOGGER.info(
                "Evaluator fallback success | study=%s | reviewer=%s | elapsed=%.2fs",
                study,
                reviewer_id,
                time.perf_counter() - started_at,
            )
            return result
        except Exception as fallback_exc:
            exc = fallback_exc
        LOGGER.exception(
            "Evaluator failed | study=%s | reviewer=%s | elapsed=%.2fs | error=%s",
            study,
            reviewer_id,
            time.perf_counter() - started_at,
            short_error_message(exc),
        )
        raise


def aggregate_rob_scores(
    config: ApiConfig,
    study: str,
    evaluator_results: list[dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    LOGGER.info("Aggregator start | study=%s", study)
    normalizer = lambda payload: normalize_aggregator_result(payload, evaluator_results)
    try:
        result = call_json_with_retry(
            config=config,
            messages=build_aggregator_messages(study, evaluator_results),
            normalizer=normalizer,
            temperature=AGGREGATOR_TEMPERATURE,
            timeout_seconds=AGGREGATOR_TIMEOUT_SECONDS,
            max_tokens=AGGREGATOR_MAX_TOKENS,
        )
        LOGGER.info(
            "Aggregator success | study=%s | elapsed=%.2fs",
            study,
            time.perf_counter() - started_at,
        )
        return result
    except Exception as exc:
        LOGGER.warning(
            "Aggregator format fallback | study=%s | error=%s",
            study,
            short_error_message(exc),
        )
        try:
            result = call_json_with_retry(
                config=config,
                messages=build_aggregator_messages(
                    study,
                    evaluator_results,
                    format_retry=True,
                    last_error=short_error_message(exc),
                ),
                normalizer=normalizer,
                temperature=AGGREGATOR_TEMPERATURE,
                timeout_seconds=FORMAT_RETRY_TIMEOUT_SECONDS,
                max_tokens=FORMAT_RETRY_MAX_TOKENS,
            )
            LOGGER.info(
                "Aggregator fallback success | study=%s | elapsed=%.2fs",
                study,
                time.perf_counter() - started_at,
            )
            return result
        except Exception as fallback_exc:
            exc = fallback_exc
        LOGGER.exception(
            "Aggregator failed | study=%s | elapsed=%.2fs | error=%s",
            study,
            time.perf_counter() - started_at,
            short_error_message(exc),
        )
        fallback_result = build_aggregator_fallback_result(evaluator_results)
        LOGGER.warning(
            "Aggregator 本地仲裁兜底 | study=%s | elapsed=%.2fs | overall=%s",
            study,
            time.perf_counter() - started_at,
            fallback_result["Overall"],
        )
        return fallback_result


def build_rob_error_row(study: str, message: str = "全文偏倚评估失败") -> dict[str, Any]:
    row = {"Study": study}
    row.update({domain: "Error" for domain in ROB_DOMAINS})
    row["_error"] = message[:200] if message else "全文偏倚评估失败"
    return row


def prepare_rob_jobs(uploaded_files: list[Any], local_text_paths: str = "") -> list[dict[str, Any]]:
    study_name_counter: Counter[str] = Counter()
    jobs = []
    next_index = 0

    for uploaded_file in uploaded_files:
        file_name = str(getattr(uploaded_file, "name", "") or "")
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".pdf", *FULLTEXT_TEXT_SUFFIXES}:
            continue

        base_study = normalize_study_name(Path(file_name).stem, f"Study_{next_index + 1}")
        study_name_counter[base_study] += 1
        suffix = study_name_counter[base_study]
        study = base_study if suffix == 1 else f"{base_study} ({suffix})"
        jobs.append(
            {
                "index": next_index,
                "study": study,
                "source_kind": "pdf" if Path(file_name).suffix.lower() == ".pdf" else "text_upload",
                "source_name": file_name,
                "suffix": Path(file_name).suffix.lower(),
                "bytes": uploaded_file.getvalue(),
            }
        )
        next_index += 1

    for path in parse_local_fulltext_paths(local_text_paths):
        base_study = normalize_study_name(path.stem, f"Study_{next_index + 1}")
        study_name_counter[base_study] += 1
        suffix = study_name_counter[base_study]
        study = base_study if suffix == 1 else f"{base_study} ({suffix})"
        jobs.append(
            {
                "index": next_index,
                "study": study,
                "source_kind": "text_path",
                "source_name": str(path),
                "suffix": path.suffix.lower(),
                "path": str(path),
            }
        )
        next_index += 1

    return jobs


def load_fulltext_job_content(
    job: dict[str, Any],
    *,
    config: ApiConfig,
    progress_queue: queue.Queue,
    use_llm_pdf_extraction: bool,
) -> tuple[str, int, str]:
    study = str(job["study"])
    source_kind = str(job.get("source_kind", "pdf"))

    if source_kind == "pdf":
        pdf_bytes = job["bytes"]
        LOGGER.info("Study start | study=%s | source=pdf | bytes=%s", study, len(pdf_bytes))
        pdf_text, page_count = extract_pdf_text(
            pdf_bytes,
            config=config,
            study=study,
            use_llm_refinement=use_llm_pdf_extraction,
            progress_queue=progress_queue,
        )
        return pdf_text, page_count, "pdf"

    if source_kind == "text_upload":
        file_bytes = job["bytes"]
        LOGGER.info(
            "Study start | study=%s | source=text_upload | bytes=%s | name=%s",
            study,
            len(file_bytes),
            job.get("source_name", ""),
        )
        text, _ = extract_text_from_preparsed_file(file_bytes, str(job.get("suffix", ".txt")))
        return text, 0, "text"

    if source_kind == "text_path":
        path = Path(str(job.get("path", ""))).expanduser()
        LOGGER.info("Study start | study=%s | source=text_path | path=%s", study, path)
        file_bytes = path.read_bytes()
        text, _ = extract_text_from_preparsed_file(file_bytes, path.suffix.lower())
        return text, 0, "text"

    raise ValueError(f"不支持的全文来源类型：{source_kind}")


def process_single_document_for_rob(
    config: ApiConfig,
    job: dict[str, Any],
    progress_queue: queue.Queue,
    use_llm_pdf_extraction: bool,
) -> dict[str, Any]:
    study = str(job["study"])
    try:
        fulltext, page_count, source_kind = load_fulltext_job_content(
            job,
            config=config,
            progress_queue=progress_queue,
            use_llm_pdf_extraction=use_llm_pdf_extraction,
        )
        evidence_text = build_rob_evidence_pack(fulltext)
        LOGGER.info(
            "Study extracted | study=%s | source=%s | pages=%s | chars=%s | evidence_chars=%s",
            study,
            source_kind,
            page_count,
            len(fulltext),
            len(evidence_text),
        )
        progress_queue.put(
            {
                "study": study,
                "stage": "extracted",
                "source_kind": source_kind,
                "pages": page_count,
                "chars": len(fulltext),
                "evidence_chars": len(evidence_text),
            }
        )
        progress_queue.put({"study": study, "stage": "collecting", "completed": 0, "total": 3})

        executor = ThreadPoolExecutor(max_workers=3)
        future_to_reviewer: dict[Any, int] = {}
        try:
            future_to_reviewer = {
                executor.submit(evaluate_single_reviewer, config, study, evidence_text, reviewer_id): reviewer_id
                for reviewer_id in range(1, 4)
            }
            evaluator_results_map: dict[int, dict[str, Any]] = {}

            for completed_count, future in enumerate(as_completed(future_to_reviewer), start=1):
                reviewer_id = future_to_reviewer[future]
                evaluator_results_map[reviewer_id] = future.result()
                LOGGER.info(
                    "Evaluator collected | study=%s | reviewer=%s | completed=%s/3",
                    study,
                    reviewer_id,
                    completed_count,
                )
                progress_queue.put(
                    {
                        "study": study,
                        "stage": "collecting",
                        "completed": completed_count,
                        "total": 3,
                    }
                )

            evaluator_results = [evaluator_results_map[reviewer_id] for reviewer_id in range(1, 4)]
        except Exception:
            for future in future_to_reviewer:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        progress_queue.put({"study": study, "stage": "aggregating"})
        final_result = aggregate_rob_scores(config, study, evaluator_results)
        progress_queue.put({"study": study, "stage": "completed"})
        LOGGER.info("Study completed | study=%s | overall=%s", study, final_result.get("Overall", ""))
        return {
            "Study": study,
            **final_result,
            "_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        message = short_error_message(exc)
        LOGGER.exception("Study failed | study=%s | error=%s", study, message)
        progress_queue.put({"study": study, "stage": "error", "message": message})
        return build_rob_error_row(study, message)


def format_rob_progress_text(total: int, completed: int, study_statuses: dict[str, str]) -> str:
    lines = [f"正在处理全文偏倚评估：{completed} / {total} 篇已完成"]
    if study_statuses:
        lines.append("")
        for study, status in list(study_statuses.items())[-min(6, len(study_statuses)):]:
            lines.append(f"- {study}：{status}")
    return "\n".join(lines)


def consume_rob_progress_events(progress_queue: queue.Queue, study_statuses: dict[str, str]) -> bool:
    changed = False

    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break

        stage = event.get("stage", "")
        if stage == "extracting_llm":
            chunk = int(event.get("chunk", 0) or 0)
            total = int(event.get("total", 0) or 0)
            if total > 0:
                status_text = f"正在执行大模型辅助PDF提取（证据聚焦）...（已完成 {chunk}/{total} 块）"
            else:
                status_text = "正在执行大模型辅助PDF提取（证据聚焦）..."
        elif stage == "extracted":
            source_kind = str(event.get("source_kind", "pdf"))
            pages = int(event.get("pages", 0) or 0)
            chars = int(event.get("chars", 0) or 0)
            evidence_chars = int(event.get("evidence_chars", 0) or 0)
            if source_kind == "text":
                status_text = f"已加载提取全文：原文约 {chars:,} 字符，评估证据约 {evidence_chars:,} 字符"
            else:
                status_text = f"PDF文本提取完成：{pages} 页，原文约 {chars:,} 字符，评估证据约 {evidence_chars:,} 字符"
        elif stage == "collecting":
            completed = int(event.get("completed", 0) or 0)
            total = int(event.get("total", 3) or 3)
            status_text = f"正在收集 3 位专家的独立评估...（已完成 {completed}/{total}）"
        elif stage == "aggregating":
            status_text = "正在汇总最终结果..."
        elif stage == "completed":
            status_text = "已完成"
        elif stage == "error":
            status_text = f"处理失败：{event.get('message', '未知错误')[:120]}"
        else:
            status_text = "处理中..."

        study = str(event.get("study", "未知文献"))
        if study in study_statuses:
            study_statuses.pop(study)
        study_statuses[study] = status_text
        changed = True

    return changed


def run_batch_rob_assessment(
    jobs: list[dict[str, Any]],
    config: ApiConfig,
    max_parallel_papers: int,
    use_llm_pdf_extraction: bool,
    progress_bar: Any,
    status_placeholder: Any,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    total = len(jobs)
    if total == 0:
        return pd.DataFrame(columns=ROB_OUTPUT_COLUMNS), []

    results: list[dict[str, Any]] = [build_rob_error_row(job["study"], "任务未执行") for job in jobs]
    progress_queue: queue.Queue = queue.Queue()
    study_statuses: dict[str, str] = {}

    progress_bar.progress(0)
    status_placeholder.info(format_rob_progress_text(total, 0, study_statuses))

    worker_count = min(max_parallel_papers, total)
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                process_single_document_for_rob,
                config,
                job,
                progress_queue,
                use_llm_pdf_extraction,
            ): idx
            for idx, job in enumerate(jobs)
        }

        pending_futures = set(future_to_index)
        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=0.2,
                return_when=FIRST_COMPLETED,
            )

            should_refresh = consume_rob_progress_events(progress_queue, study_statuses)

            for future in done_futures:
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[index] = build_rob_error_row(jobs[index]["study"], short_error_message(exc))
                completed += 1
                progress_bar.progress(completed / total)
                should_refresh = True

            if should_refresh:
                status_placeholder.info(format_rob_progress_text(total, completed, study_statuses))

    consume_rob_progress_events(progress_queue, study_statuses)

    result_df = pd.DataFrame(
        [{column: row.get(column, "Error") for column in ROB_OUTPUT_COLUMNS} for row in results],
        columns=ROB_OUTPUT_COLUMNS,
    )
    error_records = [
        {"Study": row["Study"], "Error": row.get("_error", "未知错误")}
        for row in results
        if any(row.get(domain) == "Error" for domain in ROB_DOMAINS)
    ]
    return result_df, error_records


def render_sidebar() -> tuple[ApiConfig, int]:
    st.sidebar.header("设置区")
    api_key = st.sidebar.text_input("API Key", type="password")
    base_url = st.sidebar.text_input("Base URL", value=DEFAULT_BASE_URL)
    model_name = st.sidebar.text_input("Model Name", value="gpt-4.1-mini")
    max_workers = st.sidebar.slider("最大并发数 (Max Workers)", min_value=1, max_value=20, value=5)

    config = ApiConfig(
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        model_name=model_name.strip(),
    )
    return config, max_workers


def validate_api_config(config: ApiConfig) -> bool:
    if not config.api_key:
        st.warning("请输入 API Key。")
        return False
    if not config.base_url:
        st.warning("请输入 Base URL。")
        return False
    if not config.model_name:
        st.warning("请输入 Model Name。")
        return False
    return True


def validate_screening_inputs(config: ApiConfig, topic: str, uploaded_file: Any) -> bool:
    if not validate_api_config(config):
        return False
    if not topic:
        st.warning("请输入课题描述或评估标准。")
        return False
    if uploaded_file is None:
        st.warning("请先上传 CSV 文件。")
        return False
    return True


def validate_rob_inputs(config: ApiConfig, uploaded_files: list[Any], local_text_paths: str) -> bool:
    if not validate_api_config(config):
        return False

    parsed_paths = parse_local_fulltext_paths(local_text_paths)
    if not uploaded_files and not parsed_paths:
        st.warning("请先上传至少 1 个 PDF/Markdown/TXT 文件，或填写已提取全文文件路径。")
        return False

    invalid_paths = [str(path) for path in parsed_paths if not path.exists()]
    if invalid_paths:
        st.warning("以下全文文件路径不存在：\n" + "\n".join(invalid_paths[:5]))
        return False

    unsupported_paths = [str(path) for path in parsed_paths if path.suffix.lower() not in FULLTEXT_TEXT_SUFFIXES]
    if unsupported_paths:
        st.warning("本地全文文件路径仅支持 .md / .txt：\n" + "\n".join(unsupported_paths[:5]))
        return False
    return True


def render_screening_tab(config: ApiConfig, max_workers: int) -> None:
    st.caption("上传 Zotero/EndNote 导出的 CSV，配置模型参数后即可进行批量初筛。")

    topic = st.text_area(
        "请输入当前课题的 PICO 或评估标准",
        value=DEFAULT_TOPIC,
        height=140,
        key="screening_topic",
    )
    uploaded_file = st.file_uploader("上传文献 CSV 文件", type=["csv"], key="screening_csv")
    run_clicked = st.button("开始筛选", type="primary", use_container_width=True, key="screening_run")

    if not run_clicked:
        return

    if not validate_screening_inputs(config, topic.strip(), uploaded_file):
        return

    try:
        raw_df = load_csv_file(uploaded_file)
        prepared_df = prepare_dataframe(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"文件处理失败：{exc}")
        return

    with st.spinner("正在验证 API 配置..."):
        api_error = test_api_connection(config, topic.strip())
    if api_error:
        st.error(f"API 连接测试失败：{api_error}")
        st.info(
            "请重点检查：API Key 是否正确、Base URL 是否为兼容的 `/v1` 地址、"
            "Model Name 是否存在，以及当前服务商是否支持 OpenAI Chat Completions "
            "和 `response_format={\"type\": \"json_object\"}`。"
        )
        return

    st.info(f"检测到 {len(prepared_df)} 篇有效文献，开始进行 AI 初筛。")
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    with st.spinner("模型正在并发处理文献，请稍候..."):
        result_df = run_batch_screening(
            df=prepared_df,
            config=config,
            topic=topic.strip(),
            max_workers=max_workers,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
        )

    progress_bar.progress(1.0)
    status_placeholder.success(f"处理完成: {len(result_df)} / {len(result_df)} 篇")

    st.subheader("结果预览（前 5 条）")
    st.dataframe(result_df.head(5), use_container_width=True)

    error_df = result_df[result_df["AI Decision"] == "Error"].copy()
    if not error_df.empty:
        st.warning(f"共有 {len(error_df)} 篇文献请求失败，已自动保留错误信息。")
        error_summary = (
            Counter(error for error in error_df["AI Error"].fillna("") if error.strip())
            or Counter({"未知错误": len(error_df)})
        )
        summary_df = pd.DataFrame(
            [{"错误原因": reason, "数量": count} for reason, count in error_summary.most_common()]
        )
        st.subheader("失败原因汇总")
        st.dataframe(summary_df, use_container_width=True)

        if len(error_df) == len(result_df):
            st.error("所有请求均失败，这通常说明是 API 配置或服务兼容性问题，而不是单篇文献内容问题。")

    csv_bytes = dataframe_to_csv_bytes(result_df)
    st.download_button(
        label="下载筛选结果 CSV",
        data=csv_bytes,
        file_name="meta_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_rob_tab(config: ApiConfig) -> None:
    st.caption("上传 RCT 全文 PDF，或直接提供已提取的 Markdown/TXT 全文，系统将采用 3 位独立评价员并发评估，再由 1 位资深主裁进行最终仲裁。")

    uploaded_files = st.file_uploader(
        "上传全文 PDF / Markdown / TXT 文件",
        type=FULLTEXT_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="rob_pdf_uploader",
    )
    local_text_paths = st.text_area(
        "已提取全文文件路径（可多行，支持 .md / .txt）",
        value="",
        height=100,
        key="rob_fulltext_paths",
        help="如果你已经用 MinerU 或其他工具完成全文提取，可以直接粘贴本地 .md/.txt 路径，系统会跳过 PDF 提取。",
    )
    max_parallel_papers = st.slider(
        "并发处理文章数",
        min_value=1,
        max_value=3,
        value=1,
        help="每篇文章会触发 4 次 API 调用，请根据模型配额与限流情况谨慎调整。",
    )
    use_llm_pdf_extraction = st.checkbox(
        "启用大模型辅助PDF提取",
        value=False,
        help="仅在上传 PDF 时生效。已提供 Markdown/TXT 全文时会自动跳过这一阶段。",
    )
    run_clicked = st.button(
        "开始多智能体全文偏倚评估",
        type="primary",
        use_container_width=True,
        key="rob_run",
    )

    if not run_clicked:
        return

    uploaded_file_list = list(uploaded_files or [])
    if not validate_rob_inputs(config, uploaded_file_list, local_text_paths):
        return

    jobs = prepare_rob_jobs(uploaded_file_list, local_text_paths)

    st.info(f"检测到 {len(jobs)} 篇全文材料，开始执行多智能体 RoB 2 全文偏倚评估。")
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    with st.spinner("多智能体正在阅读全文并完成偏倚评估，请稍候..."):
        result_df, error_records = run_batch_rob_assessment(
            jobs=jobs,
            config=config,
            max_parallel_papers=max_parallel_papers,
            use_llm_pdf_extraction=use_llm_pdf_extraction,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
        )

    progress_bar.progress(1.0)
    status_placeholder.success(f"处理完成: {len(result_df)} / {len(result_df)} 篇")

    st.subheader("RoB 2 最终结果")
    st.dataframe(result_df, use_container_width=True)

    if error_records:
        st.warning(f"共有 {len(error_records)} 篇全文评估失败，结果已自动填充为 `Error`。")
        error_df = pd.DataFrame(error_records)
        error_summary = Counter(record["Error"] or "未知错误" for record in error_records)
        summary_df = pd.DataFrame(
            [{"错误原因": reason, "数量": count} for reason, count in error_summary.most_common()]
        )
        st.subheader("失败原因汇总")
        st.dataframe(summary_df, use_container_width=True)
        st.subheader("失败文章明细")
        st.dataframe(error_df, use_container_width=True)

    csv_bytes = dataframe_to_csv_bytes(result_df)
    st.download_button(
        label="下载 RoB 2 结果 CSV",
        data=csv_bytes,
        file_name="rob2_fulltext_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Meta 分析 AI 辅助工具", layout="wide")

    if not ensure_dependencies():
        st.stop()

    config, max_workers = render_sidebar()

    st.title("Meta 分析 AI 辅助工具")
    st.caption("支持文献摘要初筛与多智能体全文偏倚风险评估（RoB 2）。")

    screening_tab, rob_tab = st.tabs(["文献摘要初筛", "多智能体全文偏倚评估"])

    with screening_tab:
        render_screening_tab(config, max_workers)

    with rob_tab:
        render_rob_tab(config)


if __name__ == "__main__":
    main()
