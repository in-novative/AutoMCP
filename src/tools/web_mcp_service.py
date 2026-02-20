"""
Web MCP 服务模块
从多个外部网站实时获取可用的 MCP Server 列表
使用向量相似度进行语义匹配
"""

import logging
import json
import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# 缓存配置
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DURATION_HOURS = 24  # 缓存有效期24小时


@dataclass
class MCPServerInfo:
    """MCP Server 信息"""
    name: str
    description: str
    category: str
    deployment_type: str  # 本地部署/混合部署/云服务
    rating: str  # A-优质, B-良好 等
    source_url: str
    source_name: str  # 来源网站名称
    embedding: Optional[List[float]] = field(default=None, repr=False)  # 向量嵌入
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "deployment_type": self.deployment_type,
            "rating": self.rating,
            "source_url": self.source_url,
            "source_name": self.source_name
        }
    
    def get_full_text(self) -> str:
        """获取用于嵌入的完整文本"""
        return f"{self.name} {self.description} {self.category}"


class EmbeddingService:
    """
    向量嵌入服务
    使用 OpenAI API 或本地模型生成文本嵌入
    """
    
    def __init__(self):
        self.cache: Dict[str, List[float]] = {}  # 文本 -> 向量的缓存
        self.model = "text-embedding-3-small"  # OpenAI 嵌入模型
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            向量嵌入列表
        """
        # 检查缓存
        if text in self.cache:
            return self.cache[text]
        
        try:
            # 使用 OpenAI API
            from openai import AsyncOpenAI
            from config.settings import settings
            
            # 检查配置
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not configured, using fallback embedding")
                return self._fallback_embedding(text)
            
            api_key = settings.OPENAI_API_KEY.get_secret_value()
            if not api_key or api_key == "your-api-key":
                logger.warning("OPENAI_API_KEY is empty or placeholder, using fallback embedding")
                return self._fallback_embedding(text)
            
            logger.debug(f"Calling OpenAI API for embedding, text length: {len(text)}")
            
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.OPENAI_BASE_URL
            )
            
            response = await client.embeddings.create(
                model=self.model,
                input=text[:8000]  # 限制长度
            )
            
            embedding = response.data[0].embedding
            
            # 缓存结果
            self.cache[text] = embedding
            
            logger.debug(f"Successfully got embedding, dimension: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding from OpenAI: {e}, using fallback")
            # 降级：使用简单的哈希向量
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """
        备用嵌入方案
        使用改进的语义哈希生成向量，更好地保留语义信息
        """
        import hashlib
        import re
        
        # 文本预处理
        text_lower = text.lower()
        
        # 提取关键词（简单的 TF-IDF 思想）
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fa5]+\b', text_lower)
        
        # 生成 384 维的向量（更大的维度保留更多信息）
        dim = 384
        vector = [0.0] * dim
        
        # 基于词的位置和词频构建向量
        word_weights = {}
        for word in words:
            word_weights[word] = word_weights.get(word, 0) + 1
        
        # 使用词的哈希值来影响向量
        for word, weight in word_weights.items():
            # 使用 MD5 哈希
            hash_obj = hashlib.md5(word.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # 将哈希值分布到向量中
            for i in range(min(len(hash_bytes), dim)):
                # 根据词频加权
                vector[i] += (hash_bytes[i] / 255.0 - 0.5) * weight * 0.1
        
        # 添加字符级别的特征
        for i, char in enumerate(text[:100]):  # 限制长度
            char_code = ord(char) % 256
            idx = (i * 3) % dim
            vector[idx] += (char_code / 255.0 - 0.5) * 0.01
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class MCPSource(ABC):
    """
    MCP Server 数据源抽象基类
    所有数据源必须实现此接口
    """
    
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.cache: List[MCPServerInfo] = []
        self.cache_time: Optional[datetime] = None
        self.cache_duration = timedelta(hours=CACHE_DURATION_HOURS)
        self.embedding_service = EmbeddingService()
        self.cache_file = CACHE_DIR / f"{self._sanitize_filename(name)}.json"

        # 确保缓存目录存在
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 尝试从本地文件加载缓存
        self._load_cache_from_file()

    def _sanitize_filename(self, name: str) -> str:
        """将名称转换为安全的文件名"""
        return re.sub(r'[^\w\-_.]', '_', name.lower())

    def _load_cache_from_file(self) -> None:
        """从本地文件加载缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查缓存时间
                cache_time_str = data.get('cache_time')
                if cache_time_str:
                    self.cache_time = datetime.fromisoformat(cache_time_str)

                # 检查缓存是否过期
                if self.cache_time and datetime.now() - self.cache_time < self.cache_duration:
                    # 加载服务器数据
                    servers_data = data.get('servers', [])
                    self.cache = []
                    for item in servers_data:
                        server = MCPServerInfo(
                            name=item['name'],
                            description=item['description'],
                            category=item['category'],
                            deployment_type=item['deployment_type'],
                            rating=item['rating'],
                            source_url=item['source_url'],
                            source_name=item['source_name'],
                            embedding=item.get('embedding')
                        )
                        self.cache.append(server)

                    logger.info(f"Loaded {len(self.cache)} servers from local cache ({self.name})")
                else:
                    logger.debug(f"Local cache expired for {self.name}")
        except Exception as e:
            logger.warning(f"Failed to load cache from file for {self.name}: {e}")

    def _save_cache_to_file(self) -> None:
        """保存缓存到本地文件"""
        try:
            data = {
                'cache_time': self.cache_time.isoformat() if self.cache_time else None,
                'servers': []
            }

            for server in self.cache:
                server_dict = {
                    'name': server.name,
                    'description': server.description,
                    'category': server.category,
                    'deployment_type': server.deployment_type,
                    'rating': server.rating,
                    'source_url': server.source_url,
                    'source_name': server.source_name,
                    'embedding': server.embedding
                }
                data['servers'].append(server_dict)

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved {len(self.cache)} servers to local cache ({self.name})")
        except Exception as e:
            logger.warning(f"Failed to save cache to file for {self.name}: {e}")

    @abstractmethod
    async def fetch_servers(self) -> List[MCPServerInfo]:
        """
        从数据源获取 MCP Server 列表

        Returns:
            MCP Server 信息列表
        """
        pass
    
    async def get_servers(self, force_refresh: bool = False) -> List[MCPServerInfo]:
        """
        获取 Server 列表（带内存缓存和本地文件缓存）

        缓存策略：
        1. 如果内存缓存有效（未过期），直接返回
        2. 如果内存缓存过期但本地文件缓存有效，从文件加载
        3. 如果都过期，从网络获取并更新缓存

        Args:
            force_refresh: 是否强制刷新，忽略所有缓存

        Returns:
            MCP Server 列表
        """
        # 1. 检查内存缓存是否有效
        if (not force_refresh and
            self.cache and
            self.cache_time and
            datetime.now() - self.cache_time < self.cache_duration):
            logger.debug(f"Using memory cache from {self.name}")
            return self.cache

        # 2. 如果内存缓存无效，尝试从本地文件加载（可能其他实例已更新）
        if not force_refresh and not self.cache:
            self._load_cache_from_file()
            if self.cache and self.cache_time and datetime.now() - self.cache_time < self.cache_duration:
                logger.info(f"Loaded {len(self.cache)} servers from local file cache ({self.name})")
                return self.cache

        # 3. 从网络获取
        logger.info(f"Fetching servers from network ({self.name})...")
        servers = await self.fetch_servers()
        if servers:
            # 为每个 server 生成嵌入向量
            await self._generate_embeddings(servers)
            self.cache = servers
            self.cache_time = datetime.now()
            logger.info(f"Fetched {len(servers)} servers from {self.name}")

            # 保存到本地文件
            self._save_cache_to_file()

        return self.cache
    
    async def _generate_embeddings(self, servers: List[MCPServerInfo]) -> None:
        """为所有 server 生成嵌入向量"""
        for server in servers:
            try:
                text = server.get_full_text()
                server.embedding = await self.embedding_service.get_embedding(text)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {server.name}: {e}")
                server.embedding = None
    
    def is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        return (self.cache and 
                self.cache_time and 
                datetime.now() - self.cache_time < self.cache_duration)


class MCPWorldSource(MCPSource):
    """
    MCP World 数据源
    https://www.mcpworld.com/
    """
    
    def __init__(self):
        super().__init__(
            name="MCP World",
            url="https://www.mcpworld.com/"
        )
    
    async def fetch_servers(self) -> List[MCPServerInfo]:
        """从 MCP World 网站抓取 MCP Server 列表"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.url)
                response.raise_for_status()
                
                # 解析 HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 获取纯文本内容
                text = soup.get_text(separator=' ', strip=True)
                
                # 使用正则表达式解析 MCP World 特有的格式
                servers = self._parse_mcpworld_text(text)
                
                if not servers:
                    logger.warning("Could not parse MCP World using text parser, trying fallback")
                    return await self._fallback_fetch()
                
                logger.info(f"Successfully parsed {len(servers)} servers from MCP World")
                return servers[:50]  # 限制数量
                
        except Exception as e:
            logger.error(f"Failed to fetch from {self.name}: {e}")
            return await self._fallback_fetch()
    
    def _parse_mcpworld_text(self, text: str) -> List[MCPServerInfo]:
        """
        解析 MCP World 网站的文本内容
        格式示例：
        "网页内容抓取服务器平台精选By modelcontextprotocol一个模型上下文协议服务器...A-优质本地部署"
        """
        import re
        servers = []
        
        # 匹配模式：服务名称 + 分类 + "By" + 作者/组织 + 描述 + 评级 + 部署类型
        # 示例："网页内容抓取服务器平台精选By modelcontextprotocol一个模型上下文协议服务器...A-优质本地部署"
        
        # 首先尝试查找带有评级和部署类型的完整条目
        # 模式：任意文本 + (A-优质|B-良好) + (本地部署|混合部署|云服务)
        pattern = r'([^A-Z]{3,50}?(?:服务器|工具|MCP|Server))([^A-Z]{10,200}?)(A-优质|B-良好)(本地部署|混合部署|云服务)'
        
        matches = re.findall(pattern, text)
        
        if matches:
            for match in matches:
                name_part, desc_part, rating, deployment = match
                
                # 清理名称
                name = self._extract_name_from_text(name_part)
                
                # 提取分类
                category = self._extract_category_from_text(name_part + desc_part)
                
                # 构建描述
                description = f"{name_part.strip()} {desc_part.strip()}".strip()
                
                server = MCPServerInfo(
                    name=self._sanitize_name(name),
                    description=description[:200],
                    category=category,
                    deployment_type=deployment,
                    rating=rating,
                    source_url=self.url,
                    source_name=self.name
                )
                servers.append(server)
        
        # 如果没有匹配到，尝试更宽松的匹配
        if not servers:
            # 查找包含 "By" 的条目，后面跟着作者名
            # 模式：服务描述 By 作者名
            pattern2 = r'([^B]{3,100})By\s+([a-zA-Z0-9_-]+)'
            matches2 = re.findall(pattern2, text)
            
            for desc, author in matches2:
                desc = desc.strip()
                if len(desc) < 10:
                    continue
                    
                # 提取名称（通常是描述的前部分）
                name = self._extract_name_from_text(desc)
                
                # 提取评级和部署类型
                rating = self._extract_rating_from_text(desc)
                deployment = self._extract_deployment_type_from_text(desc)
                category = self._extract_category_from_text(desc)
                
                server = MCPServerInfo(
                    name=self._sanitize_name(name),
                    description=f"{desc} By {author}",
                    category=category,
                    deployment_type=deployment,
                    rating=rating,
                    source_url=self.url,
                    source_name=self.name
                )
                servers.append(server)
        
        # 去重
        seen_names = set()
        unique_servers = []
        for server in servers:
            if server.name not in seen_names and len(server.name) > 1:
                seen_names.add(server.name)
                unique_servers.append(server)
        
        return unique_servers
    
    def _extract_name_from_text(self, text: str) -> str:
        """从文本中提取服务名称"""
        import re
        
        # 尝试匹配 "xxx服务器" 或 "xxx工具"
        patterns = [
            r'([^，。、\s]{2,20}(?:服务器|工具|MCP|Server))',
            r'^([^，。、\s]{2,30})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # 默认返回前20个字符
        return text[:20].strip()
    
    def _extract_deployment_type_from_text(self, text: str) -> str:
        """从纯文本中提取部署类型"""
        if '本地部署' in text:
            return "本地部署"
        elif '混合部署' in text:
            return "混合部署"
        elif '云服务' in text:
            return "云服务"
        return "未知"
    
    def _extract_deployment_type(self, card) -> str:
        """从卡片中提取部署类型"""
        text = card.get_text().lower()
        if '本地部署' in text or 'local' in text:
            return "本地部署"
        elif '混合部署' in text or 'hybrid' in text:
            return "混合部署"
        elif '云服务' in text or 'cloud' in text:
            return "云服务"
        return "未知"
    
    def _extract_category_from_text(self, text: str) -> str:
        """从文本中提取分类"""
        text_lower = text.lower()
        categories = {
            '浏览器自动化': ['browser', 'playwright', 'puppeteer', 'selenium', '自动化'],
            '搜索': ['search', '搜索', 'sonar', 'perplexity'],
            '数据库': ['database', 'db', 'mysql', 'postgres', 'mongo', 'redis', '数据库'],
            '文件操作': ['file', '文件', 'filesystem', 'storage'],
            '代码工具': ['code', 'git', 'github', 'documentation', '代码'],
            'AI服务': ['ai', 'llm', 'gpt', 'claude', 'openai', '人工智能'],
            '通讯': ['slack', 'discord', 'telegram', 'email', '邮件', '通讯'],
            '数据处理': ['data', 'csv', 'json', 'xml', '数据处理'],
            '可视化': ['chart', 'graph', '可视化', '图表', 'plot'],
            '系统工具': ['system', 'shell', 'command', 'terminal', '系统'],
        }
        
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return "其他"
    
    def _extract_rating_from_text(self, text: str) -> str:
        """从文本中提取评级"""
        if 'A-优质' in text or '优质' in text:
            return "A-优质"
        elif 'B-良好' in text or '良好' in text:
            return "B-良好"
        return "未评级"
    
    def _sanitize_name(self, name: str) -> str:
        """清理名称"""
        import re
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.lower()[:50]
    
    async def _fallback_fetch(self) -> List[MCPServerInfo]:
        """备用获取方案"""
        try:
            api_urls = [
                f"{self.url}api/servers",
                f"{self.url}data.json",
                f"{self.url}servers.json"
            ]
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for api_url in api_urls:
                    try:
                        response = await client.get(api_url)
                        if response.status_code == 200:
                            data = response.json()
                            return self._parse_json_data(data)
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Fallback fetch failed: {e}")
        
        return []
    
    def _parse_json_data(self, data: List[Dict]) -> List[MCPServerInfo]:
        """解析 JSON 数据"""
        servers = []
        for item in data:
            try:
                server = MCPServerInfo(
                    name=self._sanitize_name(item.get('name', 'Unknown')),
                    description=item.get('description', '')[:200],
                    category=item.get('category', '其他'),
                    deployment_type=item.get('deployment_type', '未知'),
                    rating=item.get('rating', '未评级'),
                    source_url=self.url,
                    source_name=self.name
                )
                servers.append(server)
            except Exception as e:
                logger.warning(f"Failed to parse JSON item: {e}")
        return servers


class GitHubMCPSource(MCPSource):
    """
    GitHub Awesome MCP 数据源
    从 awesome-mcp-servers README 解析 MCP 服务器列表
    """
    
    def __init__(self):
        super().__init__(
            name="GitHub Awesome MCP",
            url="https://github.com/appcypher/awesome-mcp-servers"
        )
    
    async def fetch_servers(self) -> List[MCPServerInfo]:
        """从 GitHub 获取 MCP Server 列表"""
        try:
            raw_url = "https://raw.githubusercontent.com/appcypher/awesome-mcp-servers/main/README.md"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(raw_url)
                response.raise_for_status()
                
                content = response.text
                servers = self._parse_readme(content)
                
                logger.info(f"Successfully parsed {len(servers)} servers from GitHub Awesome MCP")
                return servers
                
        except Exception as e:
            logger.error(f"Failed to fetch from {self.name}: {e}")
            return []
    
    def _parse_readme(self, content: str) -> List[MCPServerInfo]:
        """
        解析 README 内容
        区分目录链接和实际服务器条目
        """
        import re
        servers = []
        
        # 找到 "Server Implementations" 部分开始的位置
        server_section_match = re.search(r'##\s*Server Implementations', content)
        if server_section_match:
            content = content[server_section_match.start():]
        
        # 按行解析，跟踪当前分类
        lines = content.split('\n')
        current_category = "Other"
        
        for line in lines:
            line = line.strip()
            
            # 检查是否是分类标题 (## 或 ###)
            category_match = re.match(r'^#{2,3}\s*(?:<[^>]+>\s*)*(.+)$', line)
            if category_match:
                current_category = category_match.group(1).strip()
                # 清理 HTML 标签和 emoji
                current_category = re.sub(r'<[^>]+>', '', current_category)
                current_category = re.sub(r'^[^\w]*', '', current_category)  # 移除开头的非单词字符(emoji)
                continue
            
            # 解析服务器条目
            # 格式可能包含 <img> 标签: - <img .../> [name](url) - description
            # 或简单格式: - [name](url) - description
            server_match = re.match(r'-\s*(?:<[^>]+>\s*)*\[([^\]]+)\]\(([^)]+)\)\s*(?:<[^>]+>\s*)*-\s*(.+)$', line)
            if server_match:
                name, url, description = server_match.groups()
                
                # 过滤掉目录链接（URL 是锚点 #xxx）
                if url.startswith('#'):
                    continue
                
                # 过滤掉非 http 链接
                if not url.startswith('http'):
                    continue
                
                # 清理描述中的 HTML 标签
                description = re.sub(r'<[^>]+>', '', description)
                
                server = MCPServerInfo(
                    name=self._sanitize_name(name),
                    description=description.strip()[:300],
                    category=current_category,
                    deployment_type="本地部署",
                    rating="社区推荐",
                    source_url=url,
                    source_name=self.name
                )
                servers.append(server)
        
        # 去重
        seen_names = set()
        unique_servers = []
        for server in servers:
            if server.name not in seen_names and len(server.name) > 1:
                seen_names.add(server.name)
                unique_servers.append(server)

        return unique_servers
    
    def _sanitize_name(self, name: str) -> str:
        import re
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.lower()[:50]


class WebMCPService:
    """
    Web MCP 服务
    管理多个数据源，使用向量相似度进行语义搜索
    """
    
    def __init__(self):
        self.sources: List[MCPSource] = [
            # MCPWorldSource(),  # 暂时禁用：网站使用 JavaScript 动态加载，需要浏览器自动化
            GitHubMCPSource(),
        ]
        self.embedding_service = EmbeddingService()
    
    def add_source(self, source: MCPSource) -> None:
        """添加新的数据源"""
        self.sources.append(source)
        logger.info(f"Added new MCP source: {source.name}")
    
    def remove_source(self, name: str) -> bool:
        """移除数据源"""
        for i, source in enumerate(self.sources):
            if source.name == name:
                self.sources.pop(i)
                logger.info(f"Removed MCP source: {name}")
                return True
        return False
    
    def list_sources(self) -> List[str]:
        """列出所有数据源名称"""
        return [source.name for source in self.sources]
    
    async def fetch_all_servers(self, force_refresh: bool = False) -> List[MCPServerInfo]:
        """
        从所有数据源获取 MCP Server
        
        Returns:
            合并后的 MCP Server 列表（包含嵌入向量）
        """
        all_servers = []
        
        for source in self.sources:
            try:
                servers = await source.get_servers(force_refresh)
                all_servers.extend(servers)
            except Exception as e:
                logger.error(f"Failed to get servers from {source.name}: {e}")
        
        # 去重（基于名称）
        seen_names = set()
        unique_servers = []
        for server in all_servers:
            if server.name not in seen_names:
                seen_names.add(server.name)
                unique_servers.append(server)
        
        logger.info(f"Total unique servers from all sources: {len(unique_servers)}")
        return unique_servers
    
    async def search_servers(self, query: str, top_k: int = 5) -> List[Tuple[MCPServerInfo, float]]:
        """
        根据查询使用向量相似度搜索相关的 MCP Server
        
        Args:
            query: 查询文本（任务描述）
            top_k: 返回结果数量
            
        Returns:
            (MCP Server, 相似度分数) 的列表，按相似度降序排列
        """
        # 1. 获取所有 server
        servers = await self.fetch_all_servers()
        
        if not servers:
            return []
        
        # 2. 生成查询的嵌入向量
        query_embedding = await self.embedding_service.get_embedding(query)
        
        # 3. 计算与每个 server 的相似度
        scored_servers: List[Tuple[MCPServerInfo, float]] = []
        
        for server in servers:
            if server.embedding is None:
                continue
            
            similarity = self.embedding_service.cosine_similarity(
                query_embedding, 
                server.embedding
            )
            
            if similarity > 0.0:  # 只保留有相似度的结果
                scored_servers.append((server, similarity))
        
        # 4. 按相似度排序并返回 top_k
        scored_servers.sort(key=lambda x: x[1], reverse=True)
        return scored_servers[:top_k]
    
    def get_server_by_name(self, name: str) -> Optional[MCPServerInfo]:
        """根据名称获取 MCP Server"""
        for source in self.sources:
            for server in source.cache:
                if server.name == name:
                    return server
        return None
    
    def format_for_classifier(self, scored_servers: List[Tuple[MCPServerInfo, float]]) -> List[str]:
        """
        将带分数的 MCP Server 列表格式化为 Classifier 可用的字符串列表
        
        Args:
            scored_servers: (MCP Server, 相似度) 列表
            
        Returns:
            格式化的字符串列表
        """
        return [
            f"{server.name}: {server.description} "
            f"(分类: {server.category}, 相似度: {score:.2f}, 来源: {server.source_name})"
            for server, score in scored_servers
        ]
    
    async def refresh_all(self) -> Dict[str, int]:
        """刷新所有数据源"""
        stats = {}
        
        for source in self.sources:
            try:
                servers = await source.get_servers(force_refresh=True)
                stats[source.name] = len(servers)
            except Exception as e:
                logger.error(f"Failed to refresh {source.name}: {e}")
                stats[source.name] = 0
        
        return stats


# 全局单例实例
web_mcp_service = WebMCPService()
