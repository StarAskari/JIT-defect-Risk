import pandas as pd
from github import Github
import re
import os
from datetime import datetime
from tqdm import tqdm

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def is_bug_fix_commit(message):
    """
    Identify bug-fix commits using keywords
    """
    bug_keywords = [
        r'\bfix(es|ed|ing)?\b', 
        r'\bbug\b', 
        r'\bdefect\b',
        r'\bpatch\b', 
        r'\berror\b', 
        r'\bissue\s*#?\d+\b',
        r'\bresolve(d|s)?\b', 
        r'\bclose(d|s)?\s*#?\d+\b',
        r'\brepair\b',
        r'\bhotfix\b'
    ]
    message_lower = message.lower()
    return any(re.search(pattern, message_lower) for pattern in bug_keywords)

def collect_commits(repo_name, token, max_commits=5000):
    """
    Collect commits from GitHub repository
    """
    print(f"Connecting to GitHub repository: {repo_name}")
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    all_commits = []
    
    print(f"Collecting up to {max_commits} commits...")
    commit_count = 0
    
    try:
        for commit in tqdm(repo.get_commits(), total=max_commits, desc="Fetching commits"):
            if commit_count >= max_commits:
                break
            
            try:
                # Get commit details
                files = [f.filename for f in commit.files] if commit.files else []
                
                commit_data = {
                    'repo': repo_name,
                    'commit_sha': commit.sha,
                    'message': commit.commit.message,
                    'date': commit.commit.author.date,
                    'author': commit.commit.author.name if commit.commit.author else 'Unknown',
                    'files': files,
                    'files_changed': len(files),
                    'additions': commit.stats.additions,
                    'deletions': commit.stats.deletions
                }
                
                all_commits.append(commit_data)
                commit_count += 1
                
            except Exception as e:
                print(f"Error processing commit {commit.sha}: {e}")
                continue
                
    except Exception as e:
        print(f"Error fetching commits: {e}")
    
    print(f"Successfully collected {len(all_commits)} commits")
    return all_commits

def label_with_szz(all_commits):
    """
    Label commits using simplified SZZ algorithm
    """
    print("Labeling commits with SZZ algorithm...")
    
    # Step 1: Identify bug-fix commits
    bug_fix_commits = []
    for commit in all_commits:
        if is_bug_fix_commit(commit['message']):
            bug_fix_commits.append(commit)
    
    print(f"Found {len(bug_fix_commits)} bug-fix commits")
    
    # Step 2: Map bug-fixes to bug-introducing commits
    labeled_data = []
    bug_inducing_shas = set()
    
    for bug_fix in tqdm(bug_fix_commits, desc="Mapping bug-introducing commits"):
        fix_files = set(bug_fix['files'])
        
        if not fix_files:
            continue
        
        # Find the most recent previous commit that touched the same files
        for commit in all_commits:
            if commit['date'] < bug_fix['date'] and commit['commit_sha'] != bug_fix['commit_sha']:
                commit_files = set(commit['files'])
                common_files = commit_files & fix_files
                
                if common_files and commit['commit_sha'] not in bug_inducing_shas:
                    labeled_data.append({
                        'repo': commit['repo'],
                        'bug_commit_sha': commit['commit_sha'],
                        'message': commit['message'],
                        'date': commit['date'],
                        'author': commit['author'],
                        'label': 1,  # Bug-introducing
                        'fix_commit_sha': bug_fix['commit_sha'],
                        'fix_message': bug_fix['message'],
                        'files_changed': commit['files_changed'],
                        'additions': commit['additions'],
                        'deletions': commit['deletions']
                    })
                    bug_inducing_shas.add(commit['commit_sha'])
                    break  # Take only the most recent
    
    # Step 3: Add clean commits (not linked to any bug-fix and not bug-fixes themselves)
    bug_fix_shas = {bf['commit_sha'] for bf in bug_fix_commits}
    
    for commit in all_commits:
        if (commit['commit_sha'] not in bug_inducing_shas and 
            commit['commit_sha'] not in bug_fix_shas):
            labeled_data.append({
                'repo': commit['repo'],
                'bug_commit_sha': commit['commit_sha'],
                'message': commit['message'],
                'date': commit['date'],
                'author': commit['author'],
                'label': 0,  # Clean
                'fix_commit_sha': None,
                'fix_message': None,
                'files_changed': commit['files_changed'],
                'additions': commit['additions'],
                'deletions': commit['deletions']
            })
    
    print(f"Labeled {len(labeled_data)} commits")
    print(f"Bug-introducing: {sum(1 for d in labeled_data if d['label'] == 1)}")
    print(f"Clean: {sum(1 for d in labeled_data if d['label'] == 0)}")
    
    return pd.DataFrame(labeled_data)

def main():
    # Configuration
    REPO_NAME = "dotnet/runtime"
    MAX_COMMITS = 5000
    OUTPUT_FILE = "data/labeled/dotnet_runtime_labeled.csv"
    
    # Check token
    if not GITHUB_TOKEN:
        print("ERROR: GitHub token not found!")
        print("Please set GITHUB_TOKEN environment variable")
        return
    
    # Create output directory if needed
    os.makedirs("data/labeled", exist_ok=True)
    
    # Step 1: Collect commits
    all_commits = collect_commits(REPO_NAME, GITHUB_TOKEN, MAX_COMMITS)
    
    if not all_commits:
        print("No commits collected. Exiting.")
        return
    
    # Save raw commits
    raw_df = pd.DataFrame(all_commits)
    raw_df.to_csv("data/raw/dotnet_runtime_raw.csv", index=False)
    print(f"Raw commits saved to data/raw/dotnet_runtime_raw.csv")
    
    # Step 2: Label with SZZ
    labeled_df = label_with_szz(all_commits)
    
    # Step 3: Save labeled data
    labeled_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Labeled data saved to {OUTPUT_FILE}")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total commits: {len(labeled_df)}")
    print(f"Bug-introducing: {labeled_df['label'].sum()} ({labeled_df['label'].mean()*100:.2f}%)")
    print(f"Clean commits: {len(labeled_df) - labeled_df['label'].sum()} ({(1-labeled_df['label'].mean())*100:.2f}%)")
    print(f"Date range: {labeled_df['date'].min()} to {labeled_df['date'].max()}")

if __name__ == "__main__":
    main()