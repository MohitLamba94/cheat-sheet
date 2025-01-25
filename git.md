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
git fetch origin
git checkout <branch-name> 
git reset --hard origin/<branch-name>
```
If do not wish to hard reset but wish to manage conflicts use instead
```
git merge origin/<branch-name>
```


## For comparing two branches
```
https://github.com/<username>/<repository>/compare/<base-branch>...<compare-branch>
```


## General Commands
- `git status`: Tell the branch, modified files, untracked files, etc
- `git branch -a`: Lists all the local and remote branches
