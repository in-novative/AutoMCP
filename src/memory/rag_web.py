import logging
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from src.memory.rag_local import LocalToolRAG

logger = logging.getLogger(__name__)

class WebServerRAG(LocalToolRAG):
    """
    管理外部 MCP Server 的索引
    通用架构：从 URL 列表抓取 -> 解析 -> 向量化存储
    """
    
    def __init__(self, registry_urls: List[str]):
        """
        Args:
            registry_urls: 提供 MCP Server 列表的外部 URL 集合
                           (可以是 JSON API, HTML 页面, GitHub Raw 等)
        """
        super().__init__()
        # 使用独立的 Collection
        self.collection = self.client.get_or_create_collection(
            name="web_servers_v1",
            embedding_function=self.embedding_fn
        )
        self.registry_urls = registry_urls

    async def fetch_and_index(self):
        """
        [Pipeline] 1. 并发抓取 -> 2. 解析清洗 -> 3. 批量索引
        """
        if not self.registry_urls:
            logger.warning("No registry URLs configured.")
            return

        logger.info(f"Starting fetch from {len(self.registry_urls)} sources...")
        
        # 1. 并发抓取
        raw_data_list = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, url) for url in self.registry_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, res in zip(self.registry_urls, results):
                if isinstance(res, Exception):
                    logger.error(f"Failed to fetch {url}: {res}")
                elif res:
                    raw_data_list.append(res)

        # 2. 解析与清洗
        valid_servers = []
        for raw_data in raw_data_list:
            parsed = self._parse_registry_data(raw_data)
            valid_servers.extend(parsed)

        if not valid_servers:
            logger.warning("No valid servers found after parsing.")
            return

        # 3. 建立索引
        logger.info(f"Indexing {len(valid_servers)} servers...")
        self._batch_upsert(valid_servers)
        logger.info("Web RAG index updated.")

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Any:
        """通用抓取逻辑"""
        async with session.get(url, timeout=30) as response:
            response.raise_for_status()
            # 根据 Content-Type 自动决定返回 json 还是 text
            if "application/json" in response.headers.get("Content-Type", ""):
                return await response.json()
            return await response.text()

    def _parse_registry_data(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        [适配层] 将不同来源的原始数据统一转换为标准格式
        Returns: [{"name": "...", "url": "...", "description": "..."}]
        """
        servers = []
        
        # TODO: Implement your parsing logic here
        # 需要适配您实际使用的 Registry 数据格式
        # 示例逻辑：
        # if isinstance(raw_data, list): ...
        # elif isinstance(raw_data, dict) and "mcp_servers" in raw_data: ...
        
        return servers

    def _batch_upsert(self, servers: List[Dict[str, str]]):
        """底层存储操作"""
        ids = [s["name"] for s in servers]
        docs = [f"{s['name']}: {s.get('description', '')}" for s in servers]
        metas = [{"url": s["url"], "type": "web"} for s in servers]
        
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas
        )

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索最相关的 Server
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        
        output = []
        if not results["ids"]:
            return output
            
        for i, server_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            output.append({
                "name": server_id,
                "url": meta.get("url"),
                "description": results["documents"][0][i]
            })
        return output