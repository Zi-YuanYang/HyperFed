import os
import argparse

import re
import glob
import numpy as np
import scipy.io as sio
from vis_tools import Visualizer

import torch
import torch.nn as nn
import torch.optim as optim
from models import hyperfed_LEARN
import copy

from datasets import trainset_loader
from datasets import testset_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim
import time

parser = argparse.ArgumentParser()

###PDF paras
#parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=50)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--model_save_path", type=str, default="saved_models/1st")
parser.add_argument('--checkpoint_interval', type=int, default=1000000)

###federated paras
parser.add_argument("--num_clients", type = int, default = 1,help='Number of local clients')
parser.add_argument("--communication", type = int, default = 1000, help = 'Number of communications')
parser.add_argument("--epochs", type=int, default=1000, help="Number of local training")
parser.add_argument("--mode", type=str, default='hyperfed')
parser.add_argument("--mu", type=float, default=1e-6, help="the weight of fedprox")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
# visdom.Visdom(use_incoming_socket=False)
train_vis = Visualizer(env='ass')


def communication(opt, server_model, models, client_weights):
    with torch.no_grad():
        if opt.mode.lower() == 'hyperfed':
            for key in server_model.state_dict().keys():
##                if 'weight_fed' not in key and 'MLP' not in key :
#                if 'keys' not in key:
                if 'Hyper' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def my_collate(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    prj_data = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, label_data, prj_data, option, feature

def Dataset():

    src_dataset_1 = DataLoader(trainset_loader("../dataset/meta_learning/train2/geometry_1"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_2 = DataLoader(trainset_loader("../dataset/meta_learning/train2/geometry_2"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_3 = DataLoader(trainset_loader("../dataset/meta_learning/train2/geometry_3"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_4 = DataLoader(trainset_loader("../dataset/meta_learning/train2/geometry_4"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_5 = DataLoader(trainset_loader("../dataset/meta_learning/train2/geometry_5"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)
    dataloaders.append(src_dataset_5)

    return dataloaders


class net():
    def __init__(self):
#        self.model = fed_model.Learn(opt.n_block)
        self.loss = nn.MSELoss()
        self.path = opt.model_save_path
        self.train_datas = Dataset()

        self.start = 0
        self.epoch = opt.epochs
        self.com = opt.communication
        self.client_num = opt.num_clients

        self.server_model = fed_model.Learn(opt.n_block)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.server_model.to(device)


        self.models = [copy.deepcopy(self.server_model) for idx in range(self.client_num)]
        self.check_saved_model()
        #self.optimizer = optim.Adam(self.models[0].parameters(), lr=opt.lr, weight_decay=1e-8)
        self.optimizers = [torch.optim.Adam(self.models[idx].parameters(), lr=opt.lr, weight_decay=1e-8) for idx in
                           range(self.client_num)]

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/model_commu_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'model_commu_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                self.server_model.load_state_dict(torch.load(
                    '%s/model_commu_%04d.pth' % (self.path, last_epoch), map_location='cpu'))
                for wk_iter in range(self.client_num):
                    self.models[wk_iter].load_state_dict(torch.load(
                        '%s/model_worker_id(%04d)_commu_%04d.pth' % (self.path, wk_iter, last_epoch), map_location='cpu'))
                    #print(wk_iter)

    def displaywin(self, img, low=0.42, high=0.62):
        img[img<low] = low
        img[img>high] = high
        img = (img - low)/(high - low) * 255
        return img

    def initialize_weights(self):
        for module in self.server_model.modules():
            if isinstance(module, fed_model.prj_module):
                nn.init.normal_(module.weight_fed, mean=0.02, std=0.001)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
#        self.model.train(mode=True)
        for com_iter in range(self.start, self.com):
            for i_wkr in range(self.client_num):
                for epoch in range(self.epoch):
                    for batch_index, data in enumerate(self.train_datas[i_wkr]):
                        #print('456')
                        input_data, label_data, prj_data, options, feature_vec = data
                        temp = []
                        if cuda:
                            input_data = input_data.cuda()
                            label_data = label_data.cuda()
                            options = options.cuda()
                            feature_vec = feature_vec.cuda()
                            for i in range(len(prj_data)):
                                temp.append(torch.FloatTensor(prj_data[i]).cuda())
                            prj_data = temp
                        self.optimizers[i_wkr].zero_grad()
                        #self.optimizer.zero_grad()
                        output = self.models[i_wkr](input_data, prj_data, options, feature_vec)
                        loss = self.loss(output, label_data)

                        loss.backward()
                        self.optimizers[i_wkr].step()
                        #self.optimizer.step()
                        print(
                            "Com Round: %d | Worker id: %d | [Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                            % (com_iter, i_wkr, epoch + 1, self.epoch, batch_index + 1, len(self.train_datas[i_wkr]), loss.item())
                        )
                        train_vis.plot('Loss_' + str(i_wkr), loss.item())
                        train_vis.img('Ground Truth_' + str(i_wkr), self.displaywin(label_data.detach()).cpu())
                        train_vis.img('Result_' + str(i_wkr), self.displaywin(output.detach()).cpu())
                        train_vis.img('Input_' + str(i_wkr), self.displaywin(input_data.detach()).cpu())

            client_weights = [1 / self.client_num for i in range(self.client_num)]
            self.server_model, self.models = communication(opt, self.server_model, self.models, client_weights)

            if opt.checkpoint_interval != -1 and (com_iter + 1) % opt.checkpoint_interval == 0:
        # torch.save(self.models[i_wkr].state_dict(), '%s/model_com(%04d)_worker_id(%04d)_epoch_%04d.pth' % (self.path, com_iter, i_wkr, epoch + 1))
                torch.save(self.server_model.state_dict(), '%s/model_commu_%04d.pth' % (self.path, com_iter + 1))
                for check_id in range(self.client_num):
                    torch.save(self.models[check_id].state_dict(),
                            '%s/model_worker_id(%04d)_commu_%04d.pth' % (self.path, check_id, com_iter + 1))


if __name__ == "__main__":
    network = net()
    #print('4567')
    network.train()
