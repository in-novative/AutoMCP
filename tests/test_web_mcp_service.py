"""
Web MCP Service 测试脚本
测试从外部网站获取 MCP Server 列表和语义搜索功能
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.web_mcp_service import web_mcp_service, MCPServerInfo

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestWebMCP")


async def test_fetch_servers():
    """测试获取 MCP Server 列表"""
    logger.info("=" * 60)
    logger.info("测试 1: 获取所有 MCP Server")
    logger.info("=" * 60)
    
    try:
        servers = await web_mcp_service.fetch_all_servers(force_refresh=True)
        
        logger.info(f"✓ 成功获取 {len(servers)} 个 MCP Server")
        
        # 显示前 5 个
        for i, server in enumerate(servers[:5], 1):
            logger.info(f"\n{i}. {server.name}")
            logger.info(f"   描述: {server.description[:80]}...")
            logger.info(f"   分类: {server.category}")
            logger.info(f"   来源: {server.source_name}")
            logger.info(f"   是否有嵌入: {server.embedding is not None}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 获取失败: {e}")
        return False


async def test_semantic_search():
    """测试语义搜索"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 语义搜索")
    logger.info("=" * 60)
    
    test_queries = [
        "查询北京明天的天气",
        "创建一个文件并写入内容",
        "搜索最新的 AI 新闻",
        "分析 Python 代码质量",
        "发送邮件给团队成员",
        "查询数据库中的用户信息",
    ]
    
    for query in test_queries:
        logger.info(f"\n查询: '{query}'")
        logger.info("-" * 40)
        
        try:
            results = await web_mcp_service.search_servers(query, top_k=3)
            
            if results:
                for i, (server, score) in enumerate(results, 1):
                    logger.info(f"  {i}. {server.name} (相似度: {score:.3f})")
                    logger.info(f"     描述: {server.description[:60]}...")
                    logger.info(f"     分类: {server.category}")
            else:
                logger.info("  未找到相关 MCP Server")
                
        except Exception as e:
            logger.error(f"  搜索失败: {e}")


async def test_datasource_management():
    """测试数据源管理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 数据源管理")
    logger.info("=" * 60)
    
    # 列出数据源
    sources = web_mcp_service.list_sources()
    logger.info(f"✓ 当前数据源: {sources}")
    
    # 刷新统计
    logger.info("\n刷新所有数据源...")
    stats = await web_mcp_service.refresh_all()
    
    for source_name, count in stats.items():
        logger.info(f"  - {source_name}: {count} 个 Server")


async def test_format_for_classifier():
    """测试格式化输出"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: 格式化输出 (用于 Classifier)")
    logger.info("=" * 60)
    
    query = "创建一个文件并写入内容"
    logger.info(f"查询: '{query}'")
    
    try:
        results = await web_mcp_service.search_servers(query, top_k=3)
        formatted = web_mcp_service.format_for_classifier(results)
        
        logger.info("\n格式化结果:")
        for i, text in enumerate(formatted, 1):
            logger.info(f"\n{i}. {text}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 格式化失败: {e}")
        return False


async def test_embedding_service():
    """测试嵌入服务"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 5: 嵌入服务")
    logger.info("=" * 60)
    
    from src.tools.web_mcp_service import EmbeddingService
    
    service = EmbeddingService()
    
    test_texts = [
        "创建一个文件",
        "查询天气信息",
        "分析代码质量",
    ]
    
    for text in test_texts:
        logger.info(f"\n文本: '{text}'")
        
        try:
            embedding = await service.get_embedding(text)
            logger.info(f"  ✓ 嵌入维度: {len(embedding)}")
            logger.info(f"  ✓ 前 5 个值: {embedding[:5]}")
            
            # 测试相似度计算
            if text == test_texts[0]:
                embedding1 = embedding
            else:
                similarity = service.cosine_similarity(embedding1, embedding)
                logger.info(f"  ✓ 与 '{test_texts[0]}' 的相似度: {similarity:.3f}")
                
        except Exception as e:
            logger.error(f"  ✗ 失败: {e}")


async def main():
    """主测试函数"""
    logger.info("\n" + "=" * 60)
    logger.info("Web MCP Service 测试开始")
    logger.info("=" * 60)
    
    # 运行所有测试
    tests = [
        ("获取 MCP Server", test_fetch_servers),
        ("语义搜索", test_semantic_search),
        ("数据源管理", test_datasource_management),
        ("格式化输出", test_format_for_classifier),
        ("嵌入服务", test_embedding_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))
    
    # 测试总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    logger.info(f"\n总计: {passed}/{total} 通过")


if __name__ == "__main__":
    asyncio.run(main())
