import os
# 当前文件所在目录（backend/src）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（backend 的上级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 数据目录
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")