# Session Management
- Start a new session: 
  tmux new -s session_name

- Detach from a session: 
  Ctrl-b d

- List all sessions: 
  tmux ls

- Attach to an existing session: 
  tmux attach -t session_name

- Attach to the last session: 
  tmux a

- Rename a session: 
  tmux rename-session -t old_name new_name

- Kill a specific session: 
  tmux kill-session -t session_name

- Kill all sessions: 
  tmux kill-server

# Window and Pane Management
- Horizontal split: 
  Ctrl-b %

- Vertical split: 
  Ctrl-b \"

- Switch to the next pane: 
  Ctrl-b o

- Switch to a specific pane: 
  Ctrl-b q

- Resize panes: 
  Ctrl-b (arrow keys)

