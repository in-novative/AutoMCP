from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from src.memory.rag_base import BaseRAG
from config.settings import settings



class LocalToolRAG(BaseRAG):
    """
    专门管理本地工具 (LangChain Tools) 的索引
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            model_name=settings.EMBEDDING_MODEL,
            api_base=settings.OPENAI_BASE_URL
        )
        # 删除旧集合（如果存在）以确保维度一致
        try:
            self.client.delete_collection("local_tools_v1")
        except Exception:
            pass  # 集合不存在时忽略

        self.collection = self.client.get_or_create_collection(
            name="local_tools_v1", # 版本化是个好习惯
            embedding_function=self.embedding_fn
        )

    def index_item(self, item_id: str, content: str, metadata: Dict[str, Any] = None):
        self.collection.upsert(
            ids=[item_id],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def index_batch(self, items: List[Dict[str, Any]]):
        # items structure: [{"id": "tool_name", "content": "desc", "meta": {...}}]
        if not items: return
        
        self.collection.upsert(
            ids=[i["id"] for i in items],
            documents=[i["content"] for i in items],
            metadatas=[i.get("meta", {}) for i in items]
        )

    async def search(self, query: str, top_k: int = 5) -> List[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        # 返回工具名称列表
        return results["ids"][0] if results["ids"] else []

    def clear(self):
        self.client.delete_collection("local_tools_v1")