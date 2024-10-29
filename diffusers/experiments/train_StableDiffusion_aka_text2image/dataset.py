import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

'''
self.coco.keys()
dict_keys(['info', 'images', 'licenses', 'annotations'])

self.coco['annotations'][0].keys()
dict_keys(['image_id', 'id', 'caption'])

self.coco['images'][0].keys()
dict_keys(['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])

'''

'''
df = pd.read_parquet("hf://datasets/YaYaB/onepiece-blip-captions/data/train-00000-of-00001-7e86e8a67581ad82.parquet")
df.loc[0,'text']
import io
a = Image.open(io.BytesIO(df.loc[0,'image']['bytes']))
'''

class CocoDataset(Dataset):
    def __init__(self, tokeniser=None, root="/path/to/ms-coco"):
        self.tokeniser = tokeniser
        self.root = root
        with open(os.path.join(root,"annotations/captions_val2014.json"), 'r') as f:
            self.coco = json.load(f)
        self.transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return len(self.coco['annotations'])

    def __getitem__(self, index):
        # ann_id = self.coco['annotations'][index]['id']
        caption = self.coco['annotations'][index]['caption']
        img_id = self.coco['annotations'][index]['image_id']
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        path = img_info['file_name']
        
        image = Image.open(os.path.join(self.root,"val2014",path)).convert('RGB').resize((512,512),Image.LANCZOS)
        if self.transform is not None: image=self.transform(image)

        if self.tokeniser is not None: 
            caption=self.tokeniser([caption], max_length=self.tokeniser.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]        
        
        return {"pixel_values": image, "input_ids": caption}

if __name__ == "__main__":

    coco_dataset = CocoDataset()
    dataloader = DataLoader(coco_dataset, batch_size=1, shuffle=True, num_workers=0)

    for item in dataloader:
        print(item.items())
        break
