import torch.nn as nn


class WGAN_D(nn.Module):
    def __init__(self):
        super(WGAN_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(256, affine=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(512, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(1024, affine=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        # x size (1, 1, 3, 3)

        return x