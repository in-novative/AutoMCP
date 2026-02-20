import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RepoSearchResult:
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    language: str
    topics: List[str]


class GitHubRepoSearcher:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.base_url = "https://api.github.com"
    
    async def search_repos(self, query: str, limit: int = 5) -> List[RepoSearchResult]:
        try:
            headers = {
                "Accept": "application/vnd.github.v3+json"
            }
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            params = {
                "q": query,
                "per_page": limit,
                "sort": "stars",
                "order": "desc"
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/search/repositories",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("items", []):
                    results.append(RepoSearchResult(
                        name=item["name"],
                        full_name=item["full_name"],
                        description=item.get("description", ""),
                        url=item["html_url"],
                        stars=item["stargazers_count"],
                        language=item.get("language", ""),
                        topics=item.get("topics", [])
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    def format_for_classifier(self, results: List[RepoSearchResult]) -> str:
        if not results:
            return "No repositories found"
        
        lines = []
        for i, repo in enumerate(results, 1):
            lines.append(f"{i}. {repo.full_name}")
            lines.append(f"   Description: {repo.description}")
            lines.append(f"   URL: {repo.url}")
            lines.append(f"   Stars: {repo.stars} | Language: {repo.language}")
            if repo.topics:
                lines.append(f"   Topics: {', '.join(repo.topics)}")
            lines.append("")
        
        return "\n".join(lines)
