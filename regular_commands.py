'''
Set forwards hooks in PyTorch
'''
activation = {}
def getActivation(name):
    def hook(model, input, output):
        print(input.shape)
        if name not in activation:
            if torch.is_tensor(output):
                activation[name] = output.detach()
            else:
                activation[name] = output[0].detach()
    return hook
layer_interested_in = 9 
model.model.layers[layer_interested_in].self_attn.q_proj.register_forward_hook(getActivation(f"q{layer_interested_in}"))

'''
Check GPU memory usage
'''
mem_usage = [x/1e+9 for x in torch.cuda.mem_get_info(device=0)]
print("Memory consumption",mem_usage[1] - mem_usage[0])

'''
Your custom print function
'''
import builtins
f = open("tmp.txt", "w")
save_to_file = True

def print(*args):
    builtins.print(*args)
    if save_to_file:
        for strr in args:
            f.write(f"{strr} ")
        f.write("\n")
        f.flush()

'''
Helpful terminal commands
'''
rsync -av --include='july11*/' --exclude='*' /source/path login-server:/destination/path/

cp -r `ls -A | grep -v "dir2"` /home/sk/backup/
# The command lists all files and directories in the current directory, excluding “dir2”.
# It then copies all the listed files and directories (except “dir2”) recursively to /home/sk/backup/                                                    

find . -type f -name '*.jpg' -delete
# searches all its subdirectories for *.jpg and deltes them

sudo kill -9 <pid> 
ps -u -p $pid
# Tells the username

du -sh .
# disk usage

df -h
# display information about the file system disk space usage

chmod -R 777 /home/jack/


'''
Conda commands
'''
conda info --envs
conda create -p /mnt/s1/workspace/anaconda/envs/py171 --clone myenv
conda list
conda remove --name myenv --all

conda env export > environment.yaml
conda env create -f environment.yaml


'''
Pandas commands
'''
pd.read_csv(file, delimiter="|",quoting = csv.QUOTE_NONE)
# no characters will be considered as quote characters

mask = df["Column"].str.contains("id1|id2", case=False, na=False)
mask = csvFile_train["FNo"].isin([4,3])


df['Column'] = df['Column'].apply(lambda x: f"PREFIX{x}")

combine = pd.concat([df1, df2], axis=0, ignore_index=True).reset_index(drop=True)

df = df.assign(C=df['B'])
#Replicate column 'B' and assign it to a new column 'C'

df2.at[idx, "GT"] = df1.loc[num, "GT"]

df.insert(PosIndex, 'NewColumn', NewValue)

df['Column'] = df['Column'].str.replace('str1','str2')

