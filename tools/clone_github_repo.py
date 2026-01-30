import os
import subprocess
import re

from agno.tools import Toolkit
from pathlib import Path
from typing import Optional


class GitClone(Toolkit):
    def __init__(self):
        super().__init__(name="clone_github_repo", tools=[self.clone_github_repo])
        self.target_path = '/srv/AgnoCodingAgent/Knowledge/codebase'  # 直接定死，不要给模型改。建议配置绝对路径。

    def clone_github_repo(self, github_url: str) -> str:
        """
        克隆 GitHub 仓库到指定文件夹。如果没有找到本地的文件夹
        Args:
            github_url: GitHub 仓库的 URL（支持 HTTPS 和 SSH 格式）
            
        Returns:
            str: 该仓库在本地的绝对路径
            
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当克隆操作失败时
        """
        
        # 验证 GitHub URL 格式
        github_patterns = [
            r'https://github\.com/[\w-]+/[\w.-]+(?:\.git)?',
            r'git@github\.com:[\w-]+/[\w.-]+(?:\.git)?',
            r'github\.com/[\w-]+/[\w.-]+'
        ]
        
        is_valid_url = any(re.match(pattern, github_url) for pattern in github_patterns)
        if not is_valid_url:
            raise ValueError(f"无效的 GitHub URL: {github_url}")
        
        # 标准化 URL（如果没有协议前缀，添加 https://）
        if not github_url.startswith(('https://', 'git@')):
            github_url = f'https://{github_url}'
        
        # 确保 URL 以 .git 结尾（可选，但更规范）
        if not github_url.endswith('.git') and github_url.startswith('https://'):
            github_url = f'{github_url}.git'
        
        # 从 URL 中提取仓库名称
        repo_name = github_url.rstrip('/').rstrip('.git').split('/')[-1]
        
        # 创建目标路径（如果不存在）
        target_dir = Path(self.target_path).expanduser().resolve()
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"无法创建目标目录 {target_dir}: {str(e)}")
        
        # 完整的仓库本地路径
        repo_local_path = target_dir / repo_name
        
        # 检查目标路径是否已存在
        if repo_local_path.exists():
            print(
                f"目标路径已存在: {repo_local_path}。"
                f"请删除该目录或选择其他路径。"
            )
            return str(repo_local_path.absolute())
        
        # 执行 git clone 命令
        try:
            print(f"正在克隆仓库: {github_url}")
            print(f"目标路径: {repo_local_path}")
            
            result = subprocess.run(
                ['git', 'clone', github_url, str(repo_local_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5分钟超时
            )
            
            print(f"克隆成功！")
            if result.stdout:
                print(result.stdout)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("克隆操作超时（超过5分钟）")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"克隆失败: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 git 命令。请确保已安装 Git 并将其添加到系统 PATH 中。"
            )
        
        # 返回绝对路径
        return str(repo_local_path.absolute())


# 示例使用
if __name__ == "__main__":
    search = GitClone()
    # 测试示例
    try:
        # 示例 1: 克隆一个公开仓库
        repo_url = "https://github.com/old-kindhearter/AgnoCodingAgent"
        target = "/srv/AgnoCodingAgent/Knowledge/vector_db/AgnoCodingAgent"
        
        local_path = search.clone_github_repo(repo_url)
        print(f"\n✅ 仓库已克隆到: {local_path}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}") 