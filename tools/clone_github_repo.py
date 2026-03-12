import os, re
import subprocess
from urllib.parse import urlparse
from agno.tools import Toolkit
from pathlib import Path
from typing import Optional


class GitClone(Toolkit):
    def __init__(self):
        super().__init__(name="clone_github_repo", tools=[self.clone_github_repo])
        self.target_path = '/srv/AgnoCodingAgent/Knowledge/codebase'  # 直接定死，不要给模型改。建议配置绝对路径。


    def _convert_github_url(github_url):
        """
        将各种GitHub URL格式转换为HTTPS格式
        """
        # 处理SSH格式: git@github.com:user/repo.git
        ssh_pattern = r'git@github\.com:(?P<user>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?'
        ssh_match = re.match(ssh_pattern, github_url)
        
        if ssh_match:
            return f"https://github.com/{ssh_match.group('user')}/{ssh_match.group('repo')}"
        
        # 处理标准URL格式
        parsed_url = urlparse(github_url)
        
        # 如果已经是HTTPS，直接返回
        if parsed_url.scheme == 'https':
            return github_url
        
        # 处理git://或ssh://scheme
        if parsed_url.scheme in ['git', 'ssh']:
            # 移除可能的端口号和.git后缀
            netloc = parsed_url.netloc.replace(':22', '')  # 移除SSH默认端口
            path = parsed_url.path.rstrip('.git')
            return f"https://{netloc}{path}"
        
        # 其他情况返回原URL
        return github_url


    def clone_github_repo(self, github_url: str) -> str:
        """
        克隆 GitHub 仓库到指定文件夹。如果没有找到本地的文件夹
        Args:
            github_url: GitHub 仓库的 URL，如果是ssh格式则自动转换为HTTPS格式。
            
        Returns:
            str: 该仓库在本地的绝对路径
            
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当克隆操作失败时
        """
        github_url = self._convert_github_url(github_url)

        parsed_url = urlparse(github_url)
        path_parts = parsed_url.path.strip("/").split("/")
        repo_name = path_parts[-1] if path_parts else "repository"
        
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        # 3. 拼接出最终的本地绝对路径
        local_repo_path = os.path.abspath(os.path.join(self.target_path, repo_name))

        # 4. 检查目录是否已经存在
        if os.path.exists(local_repo_path) and os.listdir(local_repo_path):
            print(f"提示: 目标路径 '{local_repo_path}' 已存在且不为空，跳过 Clone 操作。")
            return local_repo_path

        # 5. 执行 git clone 命令
        try:
            print(f"正在克隆仓库: {github_url} ...")
            # subprocess.run 执行命令，check=True 表示如果命令失败则抛出异常
            subprocess.run(
                ["git", "clone", github_url, local_repo_path],
                check=True,
                text=True,
                capture_output=True # 捕获输出，避免把标准输出弄乱，如果需要看实时进度可以去掉这行
            )
            print("克隆成功！")
            return local_repo_path

        except subprocess.CalledProcessError as e:
            # 如果 clone 失败，输出错误信息
            error_msg = f"Git clone 失败!\n返回码: {e.returncode}\n错误信息: {e.stderr.strip()}"
            raise RuntimeError(error_msg)


# 示例使用
if __name__ == "__main__":
    search = GitClone()
    # 测试示例
    try:
        # 示例 1: 克隆一个公开仓库
        repo_url = "https://atomgit.com/mindspore/vllm-mindspore.git"
        
        local_path = search.clone_github_repo(repo_url)
        print(f"\n✅ 仓库已克隆到: {local_path}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}") 