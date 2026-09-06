# RAG 知识库问答系统

基于 FastAPI + Chroma + DeepSeek 构建的 RAG 知识库问答系统，支持 PDF/TXT 文档上传、混合检索、重排序、多轮对话、引用溯源和效果评估。

## 功能特点

- 📄 多格式文档支持：TXT、PDF（含表格提取、扫描件 OCR 识别）
- 🔍 混合检索：BM25 关键词检索 + 语义向量检索，加权融合
- 🎯 重排序精排：CrossEncoder 对检索结果二次精排
- 💬 多轮对话：支持查询改写，处理指代消解
- 📎 引用溯源：每条回答附原文来源和相似度分数
- 🖥️ 前后端分离：FastAPI 后端 + Streamlit 前端

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 向量库 | Chroma |
| 向量模型 | sentence-transformers |
| 检索优化 | BM25 + CrossEncoder 重排序 + RRF 融合 |
| 大模型 | DeepSeek API |
| 文档解析 | pypdf + pdfplumber + cnocr |
| 前端 | Streamlit |

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 填入 DEEPSEEK_API_KEY

# 3. 启动后端
cd backend/src
uvicorn main:app --reload

# 4. 启动前端（新终端）
streamlit run frontend/streamlit_app.py