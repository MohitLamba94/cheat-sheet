{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"CUDA_VISIBLE_DEVICES":"1"}
            "justMyCode": true,
            "cwd": "/path/to/directory/",
            "args": ["--outdir","runs/understand", "--data","/path/to/directory", "--gpus","1", "--cond","0", "--gamma=1.0", "--workers","1", "--batch","8"],
            "pythonPath": "/path/to/your/conda/env/bin/python"
        }
    ]
}
