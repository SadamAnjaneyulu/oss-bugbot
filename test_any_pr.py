import os
import sys
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_any_pr.py <owner/repo> <pr_number>")
        print("Example: python test_any_pr.py octocat/Hello-World 1")
        sys.exit(1)

    owner_repo = sys.argv[1]
    pr_number = sys.argv[2]
    
    try:
        owner, repo = owner_repo.split('/')
    except ValueError:
        print("Error: Repository must be in the format 'owner/repo' (e.g. 'pallets/flask')")
        sys.exit(1)

    print(f"Testing {owner_repo} PR #{pr_number}...")

    # Ensure required env vars are present
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is not set.")
        print("Please run: setx GITHUB_TOKEN \"your_token\" and restart your terminal.")
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("GROQ_API_KEY"):
        print("Error: Missing AI API keys in the environment.")
        sys.exit(1)

    # 1. Fetch PR details from GitHub API
    print("Fetching PR details from GitHub...")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            pr_data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch PR info. GitHub returned: {e.code} {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to fetch PR info: {e}")
        sys.exit(1)
        
    head_sha = pr_data["head"]["sha"]
    clone_url = pr_data["head"]["repo"]["clone_url"]
    print(f"Found PR HEAD SHA: {head_sha}")

    # 2. Clone the PR code to a temporary directory
    checkout_path = Path("C:/temp/pr-checkout-test").resolve()
    if checkout_path.exists():
        print("Cleaning up old checkout directory...")
        # Ignore errors to force remove even if read-only files exist
        def on_rm_error(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            os.unlink(path)
        shutil.rmtree(checkout_path, onerror=on_rm_error)
        
    print(f"Cloning the fork repository from {clone_url}...")
    subprocess.run(["git", "clone", clone_url, str(checkout_path)], check=True)
    
    print(f"Checking out the specific PR commit ({head_sha})...")
    subprocess.run(["git", "-C", str(checkout_path), "checkout", head_sha], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Set up environment variables and run main.py
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = owner_repo
    env["PR_NUMBER"] = str(pr_number)
    env["HEAD_SHA"] = head_sha
    env["PR_CHECKOUT_PATH"] = str(checkout_path)
    
    print("\n==============================================")
    print("Starting oss-bugbot pipeline...")
    print("==============================================\n")
    
    # Execute the bot
    result = subprocess.run([sys.executable, "src/main.py"], env=env)
    
    if result.returncode == 0:
        print("\n==============================================")
        print("Review complete! Check findings.json for the AI's report.")
        print("==============================================")
        
        # Optionally display the findings if there are any
        try:
            with open("findings.json", "r") as f:
                data = json.load(f)
                if data.get("findings"):
                    print(f"\nThe bot found {len(data['findings'])} bugs! Here they are:")
                    for idx, finding in enumerate(data["findings"], 1):
                        print(f"\n{idx}. {finding['title']} ({finding['file']}:{finding['line']})")
                        print(f"   Severity: {finding['severity']}")
                else:
                    print("\nThe bot found 0 bugs in this PR (either none exist, or it filtered them out).")
        except FileNotFoundError:
            pass
    else:
        print("\nAn error occurred while running the bot.")

if __name__ == "__main__":
    main()
