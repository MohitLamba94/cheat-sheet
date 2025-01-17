python -m pdb your_script.py
# start debugging from beginning

c: continue debugging until you hit a breakpoint
s: step through the code
n: to go to next line of code
l: list source code for the current file (default: 11 lines including the line being executed)
u: navigate up a stack frame
d: navigate down a stack frame
p: to print the value of an expression in the current context
pp: pretty print
where
return

# How to set the profiler
import sys
import inspect
def profile_function_calls(frame, event, arg):
    if event == "call":
        func_name = frame.f_code.co_name
        file_path = frame.f_code.co_filename
        if "/exp/root" in file_path:
            print(f"--><-- Calling {func_name} in {file_path}")
    return profile_function_calls

# Set the profiling function
sys.setprofile(profile_function_calls)

