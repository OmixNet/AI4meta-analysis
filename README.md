# Meta Analysis AI Screening Tool

基于 Streamlit 的 Meta 分析文献 AI 初筛工具。

## 功能

- 上传 Zotero / EndNote 导出的 CSV 文献数据
- 配置 API Key、Base URL、Model Name
- 并发调用大模型对标题和摘要进行初筛
- 实时显示处理进度
- 导出带有 AI 评分和判定结果的 CSV 文件

## 运行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV 要求

上传文件必须包含以下两列：

- `Title`
- `Abstract Note`

## 输出结果

程序会在原始数据基础上新增以下列：

- `AI Score`
- `AI Relevance`
- `AI Decision`
- `AI Rationale`
- `AI Error`
