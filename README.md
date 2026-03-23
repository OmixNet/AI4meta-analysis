# Meta Analysis AI Screening Tool

基于 Streamlit 的 Meta 分析文献 AI 初筛工具，用于对 Zotero / EndNote 导出的文献 CSV 进行批量标题摘要筛选，并输出结构化 AI 判定结果。

## 主要功能

- 上传 `.csv` 文献文件并自动校验关键列
- 在网页侧边栏配置 `API Key`、`Base URL`、`Model Name`
- 使用 `ThreadPoolExecutor` 进行并发调用，加快批量初筛
- 使用 `tenacity` 实现指数退避重试，提升鲁棒性
- 批量处理时实时显示进度条与状态文本
- 完成后预览结果，并导出 `utf-8-sig` 编码 CSV，避免中文乱码
- 当 API 全量失败时，提供预检报错和逐条错误原因汇总

## 目录结构

```text
.
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
└── .streamlit/
    └── config.toml
```

## 运行环境

- Python 3.10 及以上
- 支持 OpenAI 兼容接口的大模型服务

## 安装与启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

默认启动后可在本地浏览器打开 Streamlit 页面。

## 界面说明

### Sidebar

- `API Key`：大模型服务密钥，密码模式显示
- `Base URL`：默认是 `https://api.openai.com/v1`
- `Model Name`：例如 `gpt-4.1-mini`
- `Max Workers`：并发线程数，范围 1-20

### Main Area

- 输入课题的 PICO 或筛选标准
- 上传 Zotero / EndNote 导出的 CSV 文件
- 点击“开始筛选”后执行预检和批量处理

## CSV 输入要求

上传文件必须包含以下列：

- `Title`
- `Abstract Note`

程序会自动过滤掉 `Title` 与 `Abstract Note` 同时为空的无效记录。

## 输出字段

程序会在原始数据基础上新增以下列：

- `AI Score`
- `AI Relevance`
- `AI Decision`
- `AI Rationale`
- `AI Error`

其中：

- `AI Decision` 可能为 `Include`、`Unsure`、`Exclude`、`Error`
- `AI Error` 用于记录单篇请求失败的原因

## 提示词与输出格式

应用会将用户输入的课题文本拼接进系统提示词，并要求模型返回严格 JSON：

```json
{
  "score": 85,
  "relevance": "High",
  "decision": "Include",
  "rationale": "与肥胖标志物筛查高度相关"
}
```

请求中会显式设置：

```python
response_format={"type": "json_object"}
```

## 常见问题

### 1. API 请求失败

优先检查以下配置：

- `API Key` 是否有效
- `Base URL` 是否是兼容 OpenAI SDK 的 `/v1` 接口
- `Model Name` 是否存在
- 服务商是否支持 `chat.completions`
- 服务商是否支持 `response_format={"type": "json_object"}`

### 2. 所有文献都返回 Error

这通常不是文献内容问题，而是统一的接口配置或服务兼容性问题。应用会先做一次 API 预检，并在结果表中写入 `AI Error`。

### 3. 中文导出乱码

导出已使用 `utf-8-sig` 编码，可直接用 Excel 打开。

## 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub
2. 在 Streamlit Community Cloud 选择该仓库
3. 将入口文件设置为 `app.py`
4. 部署完成后在网页中填写 API 参数即可使用

## 许可证

默认使用 `MIT License`。
