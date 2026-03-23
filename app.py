import io
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            ValueError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def ensure_dependencies() -> bool:
    """检查关键依赖是否可用。"""
    missing = []
    if OpenAI is None:
        missing.append("openai")
    if retry is None:
        missing.append("tenacity")

    if missing:
        st.error(
            "缺少运行依赖："
            + ", ".join(missing)
            + "。请先安装后再运行，例如：`pip install streamlit pandas openai tenacity`"
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

    valid_mask = ~(
        (working_df["Title"] == "") &
        (working_df["Abstract Note"] == "")
    )
    filtered_df = working_df.loc[valid_mask].reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError("有效文献数为 0。所有记录的 `Title` 与 `Abstract Note` 都为空。")

    return filtered_df


def build_messages(topic: str, title: str, abstract: str) -> list[dict[str, str]]:
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


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
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


def _call_llm_once(config: ApiConfig, topic: str, title: str, abstract: str) -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai 依赖未安装。")

    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=60.0)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=build_messages(topic, title, abstract),
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = extract_message_content(response.choices[0].message.content) or "{}"
    parsed = json.loads(extract_json_text(content))
    return normalize_result(parsed)


if retry is not None:

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def call_llm_with_retry(config: ApiConfig, topic: str, title: str, abstract: str) -> dict[str, Any]:
        return _call_llm_once(config, topic, title, abstract)

else:

    def call_llm_with_retry(config: ApiConfig, topic: str, title: str, abstract: str) -> dict[str, Any]:
        return _call_llm_once(config, topic, title, abstract)


def screen_single_paper(config: ApiConfig, topic: str, title: str, abstract: str) -> dict[str, Any]:
    """单篇文献筛选，失败时返回默认结果，避免整批任务中断。"""
    try:
        return call_llm_with_retry(config, topic, title, abstract)
    except Exception as exc:  # noqa: BLE001
        return build_error_result(short_error_message(exc))


def test_api_connection(config: ApiConfig, topic: str) -> str | None:
    """批量处理前先做一次轻量 API 预检，便于提前暴露配置问题。"""
    try:
        call_llm_with_retry(
            config=config,
            topic=topic,
            title="连接测试：基于标志物重新定义肥胖的临床研究",
            abstract="本条仅用于验证 API 参数、模型名称与 JSON 输出能力是否正常。",
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


def render_main_ui() -> tuple[str, Any, bool]:
    st.title("Meta 分析文献 AI 自动化初筛助手")
    st.caption("上传 Zotero/EndNote 导出的 CSV，配置模型参数后即可进行批量初筛。")

    topic = st.text_area(
        "请输入当前课题的 PICO 或评估标准",
        value=DEFAULT_TOPIC,
        height=140,
    )
    uploaded_file = st.file_uploader("上传文献 CSV 文件", type=["csv"])
    run_clicked = st.button("开始筛选", type="primary", use_container_width=True)
    return topic.strip(), uploaded_file, run_clicked


def validate_inputs(config: ApiConfig, topic: str, uploaded_file: Any) -> bool:
    if not config.api_key:
        st.warning("请输入 API Key。")
        return False
    if not config.base_url:
        st.warning("请输入 Base URL。")
        return False
    if not config.model_name:
        st.warning("请输入 Model Name。")
        return False
    if not topic:
        st.warning("请输入课题描述或评估标准。")
        return False
    if uploaded_file is None:
        st.warning("请先上传 CSV 文件。")
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="Meta 分析文献 AI 自动化初筛助手", layout="wide")

    if not ensure_dependencies():
        st.stop()

    config, max_workers = render_sidebar()
    topic, uploaded_file, run_clicked = render_main_ui()

    if not run_clicked:
        return

    if not validate_inputs(config, topic, uploaded_file):
        st.stop()

    try:
        raw_df = load_csv_file(uploaded_file)
        prepared_df = prepare_dataframe(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"文件处理失败：{exc}")
        st.stop()

    with st.spinner("正在验证 API 配置..."):
        api_error = test_api_connection(config, topic)
    if api_error:
        st.error(f"API 连接测试失败：{api_error}")
        st.info(
            "请重点检查：API Key 是否正确、Base URL 是否为兼容的 `/v1` 地址、"
            "Model Name 是否存在，以及当前服务商是否支持 OpenAI Chat Completions "
            "和 `response_format={\"type\": \"json_object\"}`。"
        )
        st.stop()

    st.info(f"检测到 {len(prepared_df)} 篇有效文献，开始进行 AI 初筛。")
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    with st.spinner("模型正在并发处理文献，请稍候..."):
        result_df = run_batch_screening(
            df=prepared_df,
            config=config,
            topic=topic,
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


if __name__ == "__main__":
    main()
