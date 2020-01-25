import torch.nn as nn


class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()
        self.layer1 = nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.norm1 = nn.BatchNorm2d(num_features=1024)
        self.relu1 = nn.ReLU(True)
        self.layer2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=3, padding=0)
        self.norm2 = nn.BatchNorm2d(num_features=512)
        self.relu2 = nn.ReLU(True)
        self.layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.norm3 = nn.BatchNorm2d(num_features=256)
        self.layer4 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.norm3(x)
        x = self.layer4(x)
        x = self.tanh(x)
        # print("[INFO] x size:", x.size())
        x = x.view(-1, 1, 28, 28)

        return x
