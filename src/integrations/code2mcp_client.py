import os
import sys
import json
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Code2MCPResult:
    success: bool
    repo_url: str
    mcp_output_dir: Optional[str] = None
    mcp_service_path: Optional[str] = None
    service_name: Optional[str] = None
    error: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None


@dataclass
class MCPServiceInstance:
    service_name: str
    process: subprocess.Popen
    mcp_output_dir: str
    start_time: float
    port: int = 8000


class MCPServiceManager:
    def __init__(self):
        self.running_services: Dict[str, MCPServiceInstance] = {}
    
    async def start_service(self, mcp_output_dir: str, service_name: str, port: int = 8000) -> Optional[MCPServiceInstance]:
        try:
            start_script = Path(mcp_output_dir) / "start_mcp.py"
            if not start_script.exists():
                logger.error(f"Start script not found: {start_script}")
                return None
            
            env = os.environ.copy()
            env["MCP_PORT"] = str(port)
            env["MCP_TRANSPORT"] = "http"
            
            process = subprocess.Popen(
                [sys.executable, str(start_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=mcp_output_dir,
                env=env
            )
            
            await asyncio.sleep(3)
            
            if process.poll() is not None:
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"Service failed to start: {stderr}")
                return None
            
            instance = MCPServiceInstance(
                service_name=service_name,
                process=process,
                mcp_output_dir=mcp_output_dir,
                start_time=time.time(),
                port=port
            )
            
            self.running_services[service_name] = instance
            logger.info(f"Started MCP service: {service_name} on port {port}")
            return instance
            
        except Exception as e:
            logger.exception(f"Failed to start MCP service")
            return None
    
    def stop_service(self, service_name: str) -> bool:
        if service_name not in self.running_services:
            return False
        
        try:
            instance = self.running_services[service_name]
            instance.process.terminate()
            try:
                instance.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                instance.process.kill()
            
            del self.running_services[service_name]
            logger.info(f"Stopped MCP service: {service_name}")
            return True
        except Exception as e:
            logger.exception(f"Failed to stop service")
            return False
    
    def stop_all(self):
        for service_name in list(self.running_services.keys()):
            self.stop_service(service_name)
    
    def get_service(self, service_name: str) -> Optional[MCPServiceInstance]:
        return self.running_services.get(service_name)


class Code2MCPClient:
    def __init__(self, code2mcp_path: Optional[str] = None):
        self.code2mcp_path = code2mcp_path or self._detect_code2mcp_path()
        self.workspace_base = Path("./code2mcp_workspace")
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.service_manager = MCPServiceManager()
    
    def _detect_code2mcp_path(self) -> Path:
        possible_paths = [
            Path("./Code2MCP"),
            Path("../Code2MCP"),
            Path("../../Code2MCP"),
        ]
        for path in possible_paths:
            if (path / "main.py").exists():
                return path.absolute()
        
        raise FileNotFoundError("Code2MCP not found in any expected locations")
    
    async def convert_repo(self, repo_url: str, output_dir: Optional[str] = None) -> Code2MCPResult:
        try:
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            workspace_dir = self.workspace_base / repo_name
            
            if output_dir is None:
                output_dir = str(workspace_dir)
            
            cmd = [
                sys.executable,
                str(self.code2mcp_path / "main.py"),
                repo_url,
                "--output",
                str(output_dir)
            ]
            
            logger.info(f"Running Code2MCP: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.code2mcp_path)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"Code2MCP failed: {error_msg}")
                return Code2MCPResult(
                    success=False,
                    repo_url=repo_url,
                    error=error_msg
                )
            
            mcp_output_dir = Path(output_dir) / repo_name / "mcp_output"
            mcp_service_path = mcp_output_dir / "mcp_plugin" / "mcp_service.py"
            
            analysis_path = mcp_output_dir / "analysis.json"
            analysis = None
            if analysis_path.exists():
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
            
            return Code2MCPResult(
                success=True,
                repo_url=repo_url,
                mcp_output_dir=str(mcp_output_dir),
                mcp_service_path=str(mcp_service_path) if mcp_service_path.exists() else None,
                service_name=repo_name,
                analysis=analysis
            )
            
        except Exception as e:
            logger.exception("Code2MCP conversion error")
            return Code2MCPResult(
                success=False,
                repo_url=repo_url,
                error=str(e)
            )
    
    async def convert_and_start(self, repo_url: str, port: int = 8000) -> tuple[Code2MCPResult, Optional[MCPServiceInstance]]:
        conversion_result = await self.convert_repo(repo_url)
        
        if not conversion_result.success or not conversion_result.mcp_output_dir:
            return conversion_result, None
        
        service_instance = await self.service_manager.start_service(
            conversion_result.mcp_output_dir,
            conversion_result.service_name or "unknown",
            port
        )
        
        return conversion_result, service_instance
    
    def get_available_tools(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        if not analysis:
            return []
        
        tools = []
        llm_analysis = analysis.get("llm_analysis", {})
        core_modules = llm_analysis.get("core_modules", [])
        
        for module in core_modules:
            for func in module.get("functions", []):
                tools.append({
                    "name": func,
                    "description": f"{module.get('description', '')}"
                })
        
        return tools
    
    def cleanup(self):
        self.service_manager.stop_all()
