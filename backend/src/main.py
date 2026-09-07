import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from config import DOCS_DIR
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from loader import load_text_files
from splitter import  split_by_structure
from retriever import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import shutil
load_dotenv()
app = FastAPI(title="RAG 知识库问答系统", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DOCS_DIR = "D:/X/0315/data/documents"


class RAGEngine:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DOCS_DIR
        self.data_dir = data_dir
        self.store = VectorStore()
        self.store.create_collection()
        self._load_and_index()
    def _load_and_index(self):
        """增量索引：只处理新文档，已存在的文档跳过"""
        docs = load_text_files(self.data_dir)
        if not docs:
             print("⚠️ 未找到文档，跳过索引")
             return
        total_chunks = 0
        new_docs_count = 0
        for doc in docs:
            # 检查是否已索引
            if self.store.is_file_indexed(doc['filename']):
                print(f"⏭️ 跳过已索引: {doc['filename']}")
                continue
            chunks = split_by_structure(doc['content'])
            self.store.add_documents(chunks, source_file=doc['filename'])
            total_chunks += len(chunks)
            new_docs_count += 1
            print(f"✅ 已索引: {doc['filename']} ({len(chunks)} 块)")
        if new_docs_count == 0:
            print("ℹ️ 没有新文档需要索引")
        else:
            print(f"✅ 新增索引 {new_docs_count} 个文档，共 {total_chunks} 个文本块")
    def _format_history(self, history: list) -> str:
        if not history:
            return "无历史对话"
        lines = []
        for turn in history:
            lines.append(f"用户：{turn.get('user', '')}")
            lines.append(f"助手：{turn.get('assistant', '')}")
        return "\n".join(lines)
    def _rewrite_query(self, question: str, history: list = None) -> str:
        if not history:
            return question
        formatted_history = self._format_history(history)
        prompt = f"""你是一个查询改写助手。用户正在进行多轮对话，请根据对话历史，将当前问题改写为完整、独立的问句。
对话历史：
{formatted_history}

当前问题：{question}

改写后的完整问句："""

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten
    def ask(self, question: str, top_k: int = 3, history: List[dict] = None) -> dict:
        rewritten = self._rewrite_query(question, history)
        print(f"🔍 原始问题：{question}")
        print(f"📝 改写后：{rewritten}")
        results = self.store.hybrid_search(rewritten, top_k=top_k)
        if not results:
            return {"answer": "未找到相关信息", "sources": []}
        seen = set()
        unique_results = []
        for r in results:
            content = r['content'].strip()
            if content not in seen:
                seen.add(content)
                unique_results.append(r)
        context = "\n\n".join([r['content'] for r in unique_results])
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请基于参考内容回答，只说相关信息，不编造。"},
                {"role": "user", "content": f"参考内容：\n{context}\n\n问题：{question}"}
            ],
            max_tokens=500
        )
        answer = resp.choices[0].message.content
        return {
            "answer": answer,
            "sources": [{"content": r['content'], "score": r.get('rerank_score', r['score'])} for r in unique_results]
        }
rag = RAGEngine()
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    history: Optional[List[dict]] = None
class Source(BaseModel):
    content: str
    score: float
class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
@app.get("/")
def root():
    return {"message": "RAG 知识库问答系统已启动", "status": "running"}
@app.get("/status")
def status():
    return {
        "chunks_loaded": rag.store.count(),
        "vector_dim": 384
    }
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = rag.ask(request.question, request.top_k, request.history)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/upload", response_model=UploadResponse)
def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        rag._load_and_index()
        return UploadResponse(
            message="上传成功",
            filename=file.filename,
            chunks=rag.store.count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ==================== 新增流式接口 ====================
@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    def generate():
        # 1. 检索
        sources = rag.ask(request.question, request.top_k, request.history).get("sources", [])
        if not sources:
            yield "未找到相关信息"
            return
        context = "\n\n".join([s["content"] for s in sources])
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        prompt = f"""请基于参考内容回答，只说相关信息，不编造。
参考内容：
{context}

问题：{request.question}

回答："""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=500
        )
        full_answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_answer += content
                yield content
        # 发送来源信息
        yield "\n<<<SOURCES>>>\n"
        yield json.dumps({"sources": sources})
    return StreamingResponse(generate(), media_type="text/plain")
