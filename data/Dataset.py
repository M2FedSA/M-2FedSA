from torch.utils.data import Dataset
import torch
import os


class HM_Visual_Dataset(Dataset):
    def __init__(self, param):
        self.param = param
        if param == 'train':
            self.root_path = ""
        else:
            self.root_path = ""

    def __len__(self):
        count = 0
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext == '.pt':
                    count = count + 1
        return count

    def __getitem__(self, ids):
        path = self.root_path + str(ids) + '.pt'
        # print(ids)
        data = torch.load(path)
        x = data["x"]  #T,H,W,C
        y = torch.squeeze(data["y"])
        return x, y

class HM_Text_Dataset(Dataset):
    def __init__(self, param, args):
        self.param = param
        if param == 'train':
            self.root_path = ""
        else:
            self.root_path = ""


    def __len__(self):
        count = 0
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext == '.pt':
                    count = count + 1
        return count

    def __getitem__(self, ids):
        path = self.root_path + str(ids) + '.pt'
        data = torch.load(path)
        x = data["x"]  # T,H,W,C
        y = torch.squeeze(data["y"])
        return x, y

def init_dataset(name, param, args):

    Dataset_list = {}

    Dataset_list["visual"] = HM_Visual_Dataset(param)
    Dataset_list["text"] = HM_Text_Dataset(param, args)

    return Dataset_list
