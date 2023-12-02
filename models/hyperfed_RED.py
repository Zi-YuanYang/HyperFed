import torch
import torch.nn as nn
from torchsummary import summary

class RED_CNN2(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN2, self).__init__()
        self.Hyper = Hyper()

        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x, feature_vec):
        gamma, beta = self.Hyper(feature_vec)

        gamma0 = gamma[:,0:96].view(x.size(0), 96, 1, 1)
        beta0 = beta[:,0:96].view(x.size(0), 96, 1, 1)
        gamma1 = gamma[:,96:192].view(x.size(0), 96, 1, 1)
        beta1 = beta[:,96:192].view(x.size(0), 96, 1, 1)
        gamma2 = gamma[:,192].view(x.size(0), 1, 1, 1)
        beta2 = beta[:,192].view(x.size(0), 1, 1, 1)
        # encoder
        residual_1 = x
        out = self.conv1(x)
        out = gamma0 * out + beta0
        out = self.relu(out)

        out = self.conv2(out)
        out = gamma0 * out + beta0
        out = self.relu(out)
        residual_2 = out

        out = self.conv3(out)
        out = gamma0 * out + beta0
        out = self.relu(out)

        out = self.conv4(out)
        out = gamma0 * out + beta0
        out = self.relu(out)
        residual_3 = out

        out = self.conv5(out)
        out = gamma0 * out + beta0
        out = self.relu(out)
        out1 = out
        # decoder
        out = self.tconv1(out)
        out = gamma1 * out + beta1
        out += residual_3

        out = self.tconv2(self.relu(out))
        out = gamma1 * out + beta1
        out = self.tconv3(self.relu(out))
        out = gamma1 * out + beta1
        out += residual_2

        out = self.tconv4(self.relu(out))
        out = gamma1 * out + beta1
        out = self.tconv5(self.relu(out))
        out = gamma2 * out + beta2

        out += residual_1
        out = self.relu(out)
        return out


class Hyper(nn.Module):
    def __init__(self):
        super(Hyper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 386, bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        out1 = out[:,:193]
        out2 = out[:,193:]
        return out1, out2


if __name__ == '__main__':
    red = RED_CNN2()
    red.cuda()
    summary(red, input_size=(1, 128, 128))