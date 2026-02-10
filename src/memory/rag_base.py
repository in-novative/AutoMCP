from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

class BaseRAG(ABC):
    """
    RAG 服务的抽象基类
    定义了通用的增删改查接口
    """
    
    @abstractmethod
    def index_item(self, item_id: str, content: str, metadata: Dict[str, Any] = None):
        """索引单个条目"""
        pass

    @abstractmethod
    def index_batch(self, items: List[Dict[str, Any]]):
        """批量索引"""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Any]:
        """语义检索"""
        pass
    
    @abstractmethod
    def clear(self):
        """清空索引"""
        pass