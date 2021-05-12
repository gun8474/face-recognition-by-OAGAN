import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def CIR(in_c, out_c, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_c),
        nn.ReLU())
def DIR(in_c, out_c, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_c),
        nn.ReLU())

class Res_unit(nn.Module):
    def __init__(self, in_c):
        super(Res_unit, self).__init__()

        reduce_c = int(in_c / 2)
        self.layer1 = CIR(in_c, reduce_c, kernel_size=3, stride= 1, padding = 1)
        self.layer2 = CIR(reduce_c, in_c, kernel_size=3, stride= 1, padding = 1)

    def forward(self, x):
        input = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += input
        return out

class Generator_fo(nn.Module):
    def __init__(self):
        super(Generator_fo, self).__init__()

        # self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)
        #
        # self.init_size = opt.img_size // 4  # Initial size before upsampling
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.block1 = CIR(3, 64, 7, 1, 3)
        self.block2_1 = CIR(64, 128, 4, 2, 1)
        self.block2_2 = CIR(128, 256, 4, 2, 1)
        self.res = self.residual_box(256, 3)
        self.block3_1 = DIR(256, 128, 4, 2, 1)
        self.block3_2 = DIR(128, 64, 4, 2, 1)
        self.conv_block = nn.Conv2d(64, 1, 7, 1, 3)
        self.tanh = nn.Tanh()

    def residual_box(self, in_c, in_num):
        for i in range(in_num):
            box = Res_unit(in_c)
        return box

        # self.conv_blocks = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self, input):
        out = self.block1(input)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.res(out)
        out = self.block3_1(out)
        out1 = self.block3_2(out)
        out2 = self.conv_block(out1)
        out2 = self.tanh(out2)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        return out1, out2

# class Generator_fc(nn.Module):
#     def __init__(self):
#         super(Generator_fc, self).__init__()


def eltw(x, y):
    eltw = torch.matmul(x, y)
    return eltw

# def eltw_2(out1, out2):
#     m_feat = np.dot(out1, out2)
#     return m_feat

cuda = True if torch.cuda.is_available() else False
print("cuda: ", cuda)

img = torch.randn([1,3,128,128])
model = Generator_fo()
# print(np.array(model(img))[0].shape)

class FaceCompletion(nn.Module):
    def __init__(self):
        super(FaceCompletion, self).__init__()

        # 1) [conv + IN + ReLU] * 3
        # (64, 128, 128) -> (512, 16, 16)
        # instance norm에 1d를 써야하나, 2d를 써야하나? -> 1d로 하면 에러남
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU()
        )

        # 2) [deconv + IN + ReLU] * 3
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        # 3) conv + tanh
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh()
        )


    def forward(self, x):
        out=self.block1(x)
        print("1st feature map:", out.shape)

        out=self.block2(out)
        print("2nd feature map:", out.shape)

        out=self.block3(out)
        print("3rd feature map:", out.shape)
        # out=out.view(out.size(0), -1)
        # out=self.fc_layer(out)
        # print("fc layer shape:", out.shape)
        return out

# main()
model2 = FaceCompletion()

# mask : predicted occlusion mask, (1, 128, 128)
# oa_feature : OA 모듈의 output, (64, 128, 128)

# mask = torch.rand(1, 128, 128)
# print(img[0][0].view(1,1,128,128).shape) # 1채널 (1,1,128,128)
oa_feature = torch.rand(64, 128, 128)
inverse_mask = torch.ones(1,1,128,128) - model(img)[1]
occ_m = eltw(img,model(img)[1])

# occ_m = eltw(img,model(img)[0][1])
# occ_m = eltw(img,model(img)[0][2])
input_img = eltw(model(img)[0], model(img)[1])
gnn_output = model2(input_img)
g_out = eltw(gnn_output, inverse_mask)
g_out = g_out + occ_m
print(g_out.shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes + 1), nn.Softmax())

#         self.block1 = CIR(3, 64, 7, 1, 3)
#         self.block2_1 = CIR(64, 128, 4, 2, 1)
#         self.block2_2 = CIR(128, 256, 4, 2, 1)
#         self.res = self.residual_box(256, 3)
#         self.block3_1 = DIR(256, 128, 4, 2, 1)
#         self.block3_2 = DIR(128, 64, 4, 2, 1)
#         self.conv_block = nn.Conv2d(64, 1, 7, 1, 3)
#         self.tanh = nn.Tanh()

# print("mask shape: ", mask.size())
# print("oa_feature shape: ", oa_feature.size())
# print("input_img shape: ", input_img.size())  # (64, 128, 128)
# print("=================================================================")

# summary(model, (64, 128, 128))

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             """Returns layers of each discriminator block"""
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block
#
#         self.conv_blocks = nn.Sequential(
#             *discriminator_block(opt.channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )
#
#         # The height and width of downsampled image
#         ds_size = opt.img_size // 2 ** 4
#
#         # Output layers
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
#         self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes + 1), nn.Softmax())
#
#     def forward(self, img):
#         out = self.conv_blocks(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#         label = self.aux_layer(out)
#
#         return validity, label
#

# # Loss functions
# adversarial_loss = torch.nn.BCELoss()
# auxiliary_loss = torch.nn.CrossEntropyLoss()
#
# # Initialize generator and discriminator
# generator = Generator_fo()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#     auxiliary_loss.cuda()
#
# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)
#
# # Configure data loader
# os.makedirs("C:/Users/gun84/data", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "C:/Users/gun84/data",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )
#
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#
# # ----------
# #  Training
# # ----------
#
# for epoch in range(opt.n_epochs):
#     for i, (imgs, labels) in enumerate(dataloader):
#
#         batch_size = imgs.shape[0]
#
#         # Adversarial ground truths
#         valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#         fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
#         fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)
#
#         # Configure input
#         real_imgs = Variable(imgs.type(FloatTensor))
#         labels = Variable(labels.type(LongTensor))
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise and labels as generator input
#         z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
#
#         # Generate a batch of images
#         gen_imgs = generator(z) # gen이 생성한 이미지 [64,1,32,32]
#         # print(gen_imgs.data.shape[0:25])
#
#         # Loss measures generator's ability to fool the discriminator
#         validity, _ = discriminator(gen_imgs) # [64,1] # discriminator가 generator가 생성한 이미지를 예측한 것
#         g_loss = adversarial_loss(validity, valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Loss for real images
#         real_pred, real_aux = discriminator(real_imgs)
#         d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
#
#         # Loss for fake images
#         fake_pred, fake_aux = discriminator(gen_imgs.detach())
#         d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2
#
#         # Total discriminator loss
#         d_loss = (d_real_loss + d_fake_loss) / 2
#
#         # Calculate discriminator accuracy
#         pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
#         gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
#         d_acc = np.mean(np.argmax(pred, axis=1) == gt)
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
#         )
#
#         batches_done = epoch * len(dataloader) + i
#         if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
#             save_image(valid[:25], "label/%d.csv" % batches_done, nrow=5, normalize=True)
