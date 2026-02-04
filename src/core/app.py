from fastapi import FastAPI, Request, Response
from mcp.server.sse import SseServerTransport
from src.core.mcp_server import server  # 导入新的 server 实例
from config.settings import settings

app = FastAPI(title="AutoMCP", debug=settings.DEBUG)
sse = SseServerTransport("/messages")

class MCPSSEResponse(Response):
    def __init__(self, mcp_server, sse_transport, init_options):
        self.mcp_server = mcp_server
        self.sse_transport = sse_transport
        self.init_options = init_options
        super().__init__(media_type="text/event-stream")

    async def __call__(self, scope, receive, send):
        async with self.sse_transport.connect_sse(scope, receive, send) as streams:
            # Server.run 需要 read_stream, write_stream, init_options
            await self.mcp_server.run(
                streams[0], 
                streams[1], 
                self.init_options
            )

@app.get("/sse")
async def handle_sse():
    return MCPSSEResponse(
        mcp_server=server,  # 直接传入 Server 实例
        sse_transport=sse,
        init_options=server.create_initialization_options()
    )

@app.post("/messages")
async def handle_messages(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)
    return {}