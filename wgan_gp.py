import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import imageio
import os
import sys
import numpy as np
import time
import datetime
from tensorboardX import SummaryWriter
from WGAN_GP_D import WGAN_D
from WGAN_GP_G import WGAN_G


class WGAN_GP(object):
    def __init__(self, summary_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[DEVICE]", self.device.type)
        self.D = WGAN_D()
        self.G = WGAN_G()

        self.D.to(device=self.device)
        self.G.to(device=self.device)

        # WGAN values from paper
        self.lr = 2e-4
        self.b1 = 0.5
        self.b2 = 0.999

        self.num_epochs = 50
        self.batch_size = 64
        self.curr_epoch = 0

        # WGAN_gradient penalty uses Adam
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        self.latent_shape = 100
        self.critic_iter = 5
        self.curr_iter = 0
        self.gamma = 10

        self._fixed_z = torch.randn(64, self.latent_shape).to(device=self.device)

        self.writer = SummaryWriter(summary_path)
        self.sample_rate = 200

        self.images = []

        self.dataset_name = "MNIST"

    # Calculate gradient penalty
    def _gradient_penalty(self, data, generated_data):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(data)

        epsilon = epsilon.to(device=self.device)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        interpolation = interpolation.to(device=self.device)

        interpolation_logits = self.D(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        grad_outputs = grad_outputs.to(device=self.device)

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    # Training session
    def train(self):
        self._save_gif()
        start_time = time.process_time()
        data_path = os.path.join(sys.path[0], "data")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        mnist = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(mnist, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(self.num_epochs):
            print("[INFO] Starting epoch", self.curr_epoch + 1, "/", self.num_epochs + self.curr_epoch)
            self.curr_epoch += 1

            for i, data in enumerate(train_loader):
                self.curr_iter += 1
                data, _ = data
                data = data.to(self.device)
                d_loss, g_p, d_loss_real, d_loss_fake = self._train_D(data)
                W_D = d_loss_real - d_loss_fake

                print("[ITER]", self.curr_iter)
                print("[LOSS] D Loss", d_loss)
                print("[LOSS] D Real Loss", d_loss_real)
                print("[LOSS] D Fake Loss", d_loss_fake)
                print("[W_D] Wasserstein D", W_D)

                if self.curr_iter % self.critic_iter == 0:
                    g_loss = self._train_G(data.size(0))

                    print("[LOSS] G Loss", g_loss)

                    self.writer.add_scalar("g_loss", g_loss, self.curr_iter)

                if self.curr_iter % self.sample_rate == 0:
                    img_grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
                    self.writer.add_image('images', img_grid, self.curr_iter)

                self.writer.add_scalar("d_loss", d_loss, self.curr_iter)
                self.writer.add_scalar("d_loss_real", d_loss_real, self.curr_iter)
                self.writer.add_scalar("d_loss_fake", d_loss_fake, self.curr_iter)
                self.writer.add_scalar("W_D", W_D, self.curr_iter)

            save_path = "/content/drive/My Drive/tmp/WGAN_GP_adam_states/"
            self._save_state(save_path)
            model_path = "/content/drive/My Drive/tmp/WGAN_GP_adam_models/"
            self._save_model_state(model_path)

        elapsed = time.process_time() - start_time

        print("[TIME] Process Time:", str(datetime.timedelta(seconds=elapsed)))

    def train_from_checkpoint(self, checkpoint_path):
        self._load_state(checkpoint_path)
        self.train()

    # Train Discriminator
    def _train_D(self, data):
        batch_size = data.size(0)

        fake_data = self._gen_data(batch_size)

        g_p = self._gradient_penalty(data, fake_data)

        d_loss_real = self.D(data).mean()
        d_loss_fake = self.D(fake_data).mean()
        d_loss = d_loss_fake - d_loss_real + g_p

        self.d_optimizer.zero_grad()
        d_loss.backward()

        self.d_optimizer.step()
        return d_loss.item(), g_p.item(), d_loss_real.item(), d_loss_fake.item()

    # Train Generator
    def _train_G(self, batch_size):
        self.g_optimizer.zero_grad()
        fake_data = self._gen_data(batch_size)
        g_loss = -self.D(fake_data).mean()
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    # Generate data from generator
    def _gen_data(self, num):
        z = torch.randn(num, self.latent_shape)
        z = z.to(device=self.device)
        return self.G(z)

    def _save_gif(self):
        grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
        grid = np.transpose(grid.numpy(), (1, 2, 0))
        self.images.append(grid)
        imageio.mimsave('{}.gif'.format("WGAN_GP_MNIST"), self.images)

    def _save_state(self, save_path):
        save_path = save_path + self.dataset_name + '_WGAN_GP_{}.tar'.format(self.curr_epoch)
        torch.save({
            'curr_epoch': self.curr_epoch,
            'D_state_dict': self.D.state_dict(),
            'G_state_dict': self.G.state_dict(),
            'D_optim_state_dict': self.d_optimizer.state_dict(),
            'G_optim_state_dict': self.g_optimizer.state_dict(),
            'curr_iter': self.curr_iter
            }, save_path)

    def _load_state(self, load_path):
        checkpoint = torch.load(load_path)
        self.D.load_state_dict(checkpoint["D_state_dict"])
        self.G.load_state_dict(checkpoint["G_state_dict"])
        self.d_optimizer.load_state_dict(checkpoint["D_optim_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["G_optim_state_dict"])
        self.curr_iter = checkpoint["curr_iter"]
        self.curr_epoch = checkpoint["curr_epoch"]

    def _save_model_state(self, save_path):
        # Save path "/content/drive/My Drive/tmp/state_dicts/"
        file_path = (save_path + "/D/" + self.dataset_name + '_D_{}.pt'.format(self.curr_epoch))
        torch.save(self.D.state_dict(), file_path)
        print("[INFO] D state saved")

        file_path = (save_path + "/G/" + self.dataset_name + '_G_{}.pt'.format(self.curr_epoch))
        torch.save(self.G.state_dict(), file_path)
        print("[INFO] G state saved")
