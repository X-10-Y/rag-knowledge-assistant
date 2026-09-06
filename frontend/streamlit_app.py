import streamlit as st
import requests
import json
st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="📚",
    layout="wide"
)
st.title("📚 RAG 知识库问答系统")
st.markdown("上传文档，让 AI 基于文档内容回答你的问题")
API_BASE = "http://127.0.0.1:8000"
with st.sidebar:
    st.header("📤 上传文档")
    uploaded_file = st.file_uploader(
        "选择 TXT 或 PDF 文件",
        type=["txt", "pdf"],
        help="支持文本文件和 PDF 文件（含表格提取、扫描件 OCR 识别）"
    )
    if uploaded_file is not None:
        st.info(f"📄 已选择: {uploaded_file.name} ({uploaded_file.size} 字节)")
        if st.button("📥 上传并索引", type="primary"):
            with st.spinner("正在上传并建立索引..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ 上传成功！已索引 {data.get('chunks', 0)} 个文本块")
                        st.rerun()
                    else:
                        st.error(f"上传失败: {resp.text}")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")
    st.divider()
    st.subheader("📊 知识库状态")
    if st.button("刷新状态"):
        try:
            resp = requests.get(f"{API_BASE}/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.metric("文档块数量", data.get("chunks_loaded", 0))
                st.metric("向量维度", data.get("vector_dim", 0))
            else:
                st.warning("无法获取状态")
        except:
            st.warning("后端未响应，请确认服务已启动")
st.subheader("💬 提问")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 查看引用来源"):
                for i, src in enumerate(msg["sources"]):
                    st.write(f"**来源 {i+1}** (相关度: {src.get('score', 0):.3f})")
                    st.write(f"> {src.get('content', '')[:200]}...")
                    st.write("---")
if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources = []
        error_msg = None
        try:
            payload = {
                "question": prompt,
                "top_k": 3,
                "history": st.session_state.history
            }
            resp = requests.post(f"{API_BASE}/ask/stream", json=payload, stream=True, timeout=120)
            if resp.status_code == 200:
                # 累积所有接收到的数据
                full_response = ""
                for chunk in resp.iter_content(decode_unicode=True):
                    if chunk:
                        full_response += chunk
                # 查找分隔符
                delimiter = "\n<<<SOURCES>>>\n"
                if delimiter in full_response:
                    answer_part, json_part = full_response.split(delimiter, 1)
                    full_answer = answer_part.strip()
                    # 解析 JSON
                    if json_part.strip():
                        try:
                            data = json.loads(json_part.strip())
                            sources = data.get("sources", [])
                        except json.JSONDecodeError:
                            sources = []
                else:
                    # 没有分隔符，全部当做回答
                    full_answer = full_response.strip()
                # 显示回答
                placeholder.markdown(full_answer)
            else:
                error_msg = f"请求失败，状态码: {resp.status_code}"
        except requests.exceptions.ConnectionError:
            error_msg = "无法连接到后端服务，请确认后端已启动（端口 8000）"
        except Exception as e:
            error_msg = f"发生错误: {e}"
        if error_msg:
            st.error(error_msg)
        else:
            # 保存历史
            st.session_state.history.append({"user": prompt, "assistant": full_answer})
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })
            # 显示来源（如果没有回答内容但有来源，仍显示）
            if sources:
                with st.expander("📎 查看引用来源"):
                    for i, src in enumerate(sources):
                        st.write(f"**来源 {i+1}** (相关度: {src.get('score', 0):.3f})")
                        st.write(f"> {src.get('content', '')[:200]}...")
                        st.write("---")