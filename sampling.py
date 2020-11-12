import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch


tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
])

class Mydate(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        cls = torch.tensor([int(strs[1])],dtype=torch.float32)
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])],dtype=torch.float32)
        img_data = tf(Image.open(img_path))
        # img_data = torch.tensor(np.array(Image.open(img_path)) / 255. - 0.5, dtype=torch.float32)
        # img_data = img_data.permute(2,0,1)

        return img_data, cls, offset

if __name__ == '__main__':
    path = r"D:\PycharmProject\MTCNN\face_detection\celeba\12"
    datas = Mydate(path)
    data = data.DataLoader(datas,10,shuffle=False)
    for x,y,z in data:
        print(x.shape)
        print(y.shape)
        print(z.shape)

