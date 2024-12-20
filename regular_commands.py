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
Put text using PIL with white box
'''
from PIL import Image, ImageDraw, ImageFont
image = Image.open(path)
draw = ImageDraw.Draw(image)
text = f"Some text goes here"
font = ImageFont.truetype("FreeMono.ttf", 64)
bbox = draw.textbbox((0, 0), text, font=font)
text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
x, y = 10, 10  # 10 pixels from the right edge
draw.rectangle([x, y, x + text_width, y + text_height], fill="white")
draw.text((x, y), text, font=font, fill="black")
image.save("img.jpg")

'''
Helpful terminal commands
'''
grep -rl "example" /home/user/documents
# find all files in the /home/user/documents directory that contain the word "example"

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

watch -n 1 nvidia-smi
# displays nvidia-smi every 1 sec

/path/to/train.sh >> /path/to/log 2>&1
# this command runs the train.sh script and appends both its standard output and error messages to the log file. If file does not exist it will create it.


'''
Conda commands
'''
conda info --envs
conda create -p /anaconda/envs/py171 --clone myenv
conda list
conda remove --name myenv --all

conda env export > environment.yaml
conda env create -f environment.yaml

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# where cuda needs to match torch.version.cuda
# For updated list see https://anaconda.org/nvidia/cuda-toolkit


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

'''
HuggingFace
'''
huggingface-cli download bert-base-uncased
# execute this command in terminal to download the specified HF model

'''
Python House Keeping
'''
shutil.rmtree(title,ignore_errors=True)

class Example:
    def __init__(self, x, y):
        self.x = x
        self.y = y

obj = Example(10, 20)

print("dir(obj):", dir(obj))
print("vars(obj):", vars(obj))
#dir(obj): ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'x', 'y']
#vars(obj): {'x': 10, 'y': 20}


'''
How to set up a infinite dataloader in PyTorch
'''
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

self.dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

dl = cycle(self.dl)
for ind in range(self.num_train_steps):
    data = next(dl)
    # inputs, targets = data
    # outputs = model(inputs)
    # loss = criterion(outputs, targets)
    # loss.backward()
    # optimizer.step()

'''
How to convert a batch(or more dim!) of PyTorch Images to a single large image
something like torchutils.make_grid() but with more flexibity in dimensions
'''
images.shape = (16,3,32,32)

from einops import rearrange, repeat
image = rearrange(images, '(row col) c h w -> c (row h) (col w)', row = desired_number_of_rows)



