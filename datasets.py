
import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset


class trainset_loader(Dataset):
    def __init__(self, root):
        self.file_path = 'input'
        self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace(self.file_path, 'label')
        file_C = file_A.replace('input', 'projection')
        file_D = file_A.replace('input', 'geometry')
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']
        geometry = scio.loadmat(file_D)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)

        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0, 1, 4, 5, 7, 8, 10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5, 0, 0.005 - 0.001, 0.004 - 0.0015, 2.0 - 0.5, 2.0 - 0.5, 3.5])
        maxVal = torch.FloatTensor([11, 4, 0.012 + 0.001, 0.014 + 0.0015, 5.0 + 0.5, 5.0 + 0.5, 6.5])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, label_data, prj_data, option, feature

    def __len__(self):
        return len(self.files_A)


class testset_loader(Dataset):
    def __init__(self, root):
        self.files_A = []
        for i in range(0, 5):
            root_path = root + '_' + str(i + 1)
            # root_path = root
            path = os.path.join(root_path, 'test', 'input', 'data')
            # print(path)
            #        print(path)
            self.files_A = self.files_A + sorted(glob.glob(path + '*.mat'))

        self.gemoetry = torch.FloatTensor([
            [512, 368, 256, 256, 0.0133, 0.025716, 0.012268, 5.95, 4.906, 0, 0.5e5],
            [512, 315, 256, 256, 0.014, 0.03, 0.012268, 4.5, 3.5, 0, 5e5 * 0.1375],
            [384, 330, 256, 256, 0.0139, 0.026, 0.0164, 4, 3, 0, 5e5 * 0.175],
            [400, 350, 256, 256, 0.012, 0.022, 0.0157, 4, 3.5, 0, 5e5 * 0.2125],
            [384, 350, 256, 256, 0.014, 0.025, 0.0164, 5, 3, 0, 5e5 * 0.25]
        ])

    def __getitem__(self, index):
        file_A = self.files_A[index]
        res_name = file_A
        file_C = file_A.replace('input', 'projection')
        geometry_idx = int(file_A[44])

        input_data = scio.loadmat(file_A)['data']
        prj_data = scio.loadmat(file_C)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)
        geometry = self.gemoetry[geometry_idx - 1]
        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0, 1, 4, 5, 7, 8, 10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5, 0, 0.005 - 0.001, 0.004 - 0.0015, 2.0 - 0.5, 2.0 - 0.5, 3.5])
        maxVal = torch.FloatTensor([11, 4, 0.012 + 0.001, 0.014 + 0.0015, 5.0 + 0.5, 5.0 + 0.5, 6.5])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, prj_data, res_name, option, feature

    def __len__(self):
        return len(self.files_A)


class testset_loader_w_label(Dataset):
    def __init__(self, root):
        self.files_A = []
        self.label_pth = "../dataset/different gemotries/test/label/"
        for i in range(0, 5):
            root_path = root + '_' + str(i + 1)
            # root_path = root
            path = os.path.join(root_path, 'test', 'input', 'data')
            # print(path)
            #        print(path)
            self.files_A = self.files_A + sorted(glob.glob(path + '*.mat'))
        self.gemoetry = torch.FloatTensor([
            [512, 368, 256, 256, 0.0133, 0.025716, 0.012268, 5.95, 4.906, 0, 0.5e5],
            [512, 315, 256, 256, 0.014, 0.03, 0.012268, 4.5, 3.5, 0, 5e5 * 0.1375],
            [384, 330, 256, 256, 0.0139, 0.026, 0.0164, 4, 3, 0, 5e5 * 0.175],
            [400, 350, 256, 256, 0.012, 0.022, 0.0157, 4, 3.5, 0, 5e5 * 0.2125],
            [384, 350, 256, 256, 0.014, 0.025, 0.0164, 5, 3, 0, 5e5 * 0.25]
        ])

    def __getitem__(self, index):
        file_A = self.files_A[index]
        res_name = file_A[-13:]
        file_C = file_A.replace('input', 'projection')
        geometry_idx = int(file_A[44])

        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(self.label_pth + res_name)['data']
        prj_data = scio.loadmat(file_C)['data']

        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)
        geometry = self.gemoetry[geometry_idx - 1]
        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0, 1, 4, 5, 7, 8, 10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5, 0, 0.005 - 0.001, 0.004 - 0.0015, 2.0 - 0.5, 2.0 - 0.5, 3.5])
        maxVal = torch.FloatTensor([11, 4, 0.012 + 0.001, 0.014 + 0.0015, 5.0 + 0.5, 5.0 + 0.5, 6.5])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, prj_data, label_data, file_A, option, feature

    def __len__(self):
        return len(self.files_A)

