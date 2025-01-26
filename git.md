## How to create a new branch and push to remote
```
git checkout -b <new-branch-name>
git add .
git commit -m "Describe your changes here"
git push origin <new-branch-name>
```
Remove the first line if do not wish to create a new branch

## Update local branch with the latest changes from the remote branch
```
git checkout <branch-name>
```
```
git fetch origin
git merge origin/<branch-name>
```
The above can be combined into a single command
```
git pull origin <branch-name>
```
Fetch is preferred because it allows comparison before merging the remote into the local
```
git diff <branch-name> origin/<branch-name>
```
If do not wish to manage conflicts and prefer a `hard reset` of remote into local use below instead of `git merge`
```
git reset --hard origin/<branch-name>
```


## For comparing two branches
```
https://github.com/<username>/<repository>/compare/<base-branch>...<compare-branch>
```


## General Commands
- `git status`: Tell the branch, modified files, untracked files, etc
- `git branch -a`: Lists all the local and remote branches
- `git log`: Lists the commit history
- Any file can have following 4 states
  - Untracked: The file is new and not being tracked by Git.
  - Unstaged (Modified): The file has been modified in the working directory but has not yet been added to the staging area.
  - Staged: The file has been added to the staging area and is ready to be committed.
  - Committed: The changes in the file have been committed to the local repository.
 
## git reset
```
git reset [commit]
```
- It resets the HEAD to the specified commit.
- It resets the staging area (index) to match the specified commit.
- It does not touch the working directory. This means any changes you have made to the files in your working directory will remain.

```
git reset --hard [commit]
```
In addition to above
- It also resets the working directory to match the specified commit.
- This means any changes in the working directory and the staging area will be discarded and cannot be recovered.

## How to make `Feature` branch the `main` branch, rename existing `main` to `OldFeature` and push everything to remote 
```
git checkout Feature

# Rename the existing main branch to OldFeature
git branch -m main OldFeature

# Rename the Feature branch to main
git branch -m Feature main

# Push the new main branch to the remote? Can this cause Errors???
git push origin main

# Push the OldFeature branch to the remote repository
git push origin OldFeature

```
And perhaps we should use below to avoid errors
```
git push -f origin main
```

