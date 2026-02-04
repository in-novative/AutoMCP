import uvicorn
from config.settings import settings
from config.logging import setup_logging

def main():
    # 1. 初始化日志配置
    # 必须在最开始调用，确保 uvicorn 和 app 的日志格式统一
    setup_logging()
    
    # 2. 启动服务器
    # 使用字符串引用 "src.server.app:app" 支持热重载 (reload)
    # 如果生产环境不需要 reload，也可以直接传 app 对象
    uvicorn.run(
        "src.server.app:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,  # 开发模式下开启热重载
        log_config=None         # 禁用 uvicorn 默认日志配置，使用我们自定义的
    )

if __name__ == "__main__":
    main()