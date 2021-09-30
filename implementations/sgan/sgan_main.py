# 새로운 참고링크: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/context_encoder
# 우리랑 같은 3채널에 하고자하는 바도 비슷함. shape 참고하기 좋을듯하여 첨부함

# 나(소현)는 사진 11개 -> batch_size=11, num_classes=11이니 주의바람!

import argparse
import os
import numpy as np
from dataloader import OAGandataset
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
#from auxiliary_training import *
from loss import sganloss

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=11, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=11, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
# print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

# 코드 출처 : https://dnddnjs.github.io/cifar10/2018/10/09/resnet/
# https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/srgan/models.py#L18
# 여길보니 residual block 할때 in, out channel이 동일함.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out

# 참고링크: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/cogan.py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # TODO : 밑에 3줄이 의미하는 것 찾아 수정 or 삭제하기
        # self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        # self.init_size = opt.img_size // 4  # Initial size before upsampling
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.FaceOcclusion_1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # -----
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # -----
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            # -----
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
            # -----
        )
        self.FaceOcclusion_2=nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        self.FaceCompletion=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # -----
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # -----
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # occlusion aware module
        out_predicted=self.FaceOcclusion_1(x)
        out_predictedM=self.FaceOcclusion_2(out_predicted)
        # TODO: 아랫줄 1-x가 아니라 1-out_predictedM같은데 어떻게 생각하쇼?일단 바꾸겠음! -> 답변: 마자용!!!!
        out_InvertedM = torch.ones(1, 1, 128, 128) - out_predictedM
        out_oa=torch.matmul(out_predicted, out_predictedM)

        # face completion module
        out_synth=self.FaceCompletion(out_oa)
        out_fc=torch.matmul(out_InvertedM, out_synth)
        out_filter=torch.matmul(x, out_predictedM)
        out_final=out_filter + out_fc

        return out_predictedM,  out_InvertedM, out_synth, out_final


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # The height and width of downsampled image
        # ds_size = opt.img_size // 2 ** 4
        # Output layers
        # https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/network.py#L104- patch gan discriminator code
        # 기존 sgan코드는 linear였지만 우리는 논문에 따라 conv를 취하면서 shape이 달라지게 된듯.
        self.adv_layer = nn.Sequential(nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1),
                                       nn.Sigmoid())
        self.attr_layer = nn.Sequential(nn.Conv2d(2048, opt.num_classes, kernel_size=2, stride=1, padding=0),
                                        nn.Softmax())  # attribute classification대신 얼굴 인식 수행
    def forward(self, x):
        out = self.discriminator_block(x) # torch.Size([11, 2048, 2, 2])
        # out = out.view(out.shape[0], -1) # torch.Size([11, 8192])
        validity = self.adv_layer(out) # torch.Size([11, 1, 2, 2])
        label = self.attr_layer(out) # torch.Size([11, 11, 1, 1])
        # label = label.view(label.shape[0], -1) # torch.Size([11, 11]) # 왜 label view 했는지 설명바람!

        return validity, label

#TODO: loss에 할당되는 weight parameter (조정 필요 -> 14주차 발표 참고)
class weight():

    def __init__(self):
        self.lam1 = 0.2
        self.lam2 = 0.2
        self.lam3 = 0.2
        self.lam4 = 0.2
        self.lam5 = 0.1
        self.lam6 = 0.1
        self.alpha = 0.7
        self.beta = 0.3

w = weight()

# Loss functions - TODO: 지혜,승건 수정 부분
# loss 합치는거 그냥 sum of scala vector*loss 로 하면될듯?
# 참고링크: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/cogan.py 210줄
adversarial_loss = torch.nn.BCELoss()
attribute_loss = nn.MSELoss()  # discriminator에 사용되는 attribute loss

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    attribute_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# data loader
'''
data load할 부분 Index
10000장에서 7000장은 train에 사용,
그 7000장을 1000장씩 나눠서 alternative train에 사용

총 7번의 alternative train paired:unpaired 비율은 각각
9:1, 8:2, 7:3, 6:4, 5:5, 4:6,3:7

알아서 계산해서 index바꾸시길~

'''
idx1 = 0
idx2 = 899
idx3 = 900
idx4 = 999

#TODO: model save후 불러오기 (일단은 save만이라도)
#처음 안 사실 : 숫자 Parameter가 문자 parameter보다 먼저와야함..
paired_dataset = OAGandataset(idx1, idx2, paired=True, folder_numbering=False)
unpaired_dataset = OAGandataset(idx3, idx4, unpaired=True, folder_numbering=False)

train_dataloader_p = DataLoader(paired_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=opt.batch_size)

train_dataloader_up = DataLoader(unpaired_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=opt.batch_size)
print ("data loaded")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ----------
#  Training
#  TODO: alternating training 보고 디자인하기
# ----------

#나도 TODO넣고싶은데 어케함??????? -> 주석에 영어로 투두쓰면 바로 적용됩니다~
#paired image training (unpaired도 따로 만들고, loss도 상황에 따라 적용)
print ("paired train")
for epoch in range(opt.n_epochs):
    for i, (imgs,imgs_gt,labels) in enumerate(train_dataloader_p):
        batch_size = opt.batch_size

        # Adversarial ground truths
        # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        valid = Variable(FloatTensor(batch_size, 1, 2, 2).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1, 2, 2).fill_(0.0), requires_grad=False)
        # fake_attr_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)
        fake_attr_gt = Variable(FloatTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        # TODO: labels는 float 형태일 수 없음. 무조건 long type이어야함. 다른 곳에서 문제가 있는거임. -> 여러 자료 찾아봤지만 다들 이유는 모르지만 label을 float로 형변환하라고 함
        # labels = Variable(labels.type(LongTensor))
        labels = Variable(labels.type(FloatTensor))

        # line 280, line 286 -> FloatTensor가 기대된다고 해서 LongTensor -> FloatTensor 로 바꿔봄 => 에러 안남

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))) -> 우리는 사용X

        # Generate a batch of images
        out_predictedM, out_InvertedM, out_synth, out_final = generator(real_imgs) # discriminator와 loss 계산에 쓰이는 애들

        loss = sganloss([out_final,
                         out_predictedM,
                         out_InvertedM,
                         out_synth],
                        imgs_gt)
        # # Loss measures generator's ability to fool the discriminator


        validity, _ = discriminator(out_final) # ?????????? : 해결했으니까 물음표 치우시길!
        # print('validity', validity.shape) # validity torch.Size([10, 1, 2, 2])
        # print('val', valid.shape) # val torch.Size([10, 1])

        g_loss = 0
        g_loss += w.lam1*loss.perceptual_loss()
        g_loss += w.lam2*loss.style_loss()
        g_loss += w.lam3*loss.pixel_loss(w.alpha, w.beta)  
        g_loss += w.lam4*loss.smooth_loss()
        g_loss += w.lam5*loss.l2_norm()
        g_loss += w.lam6*adversarial_loss(validity,valid)
        
        print ("loss:",g_loss)
        
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # d_alpha, d_beta는 discriminator에 사용되는 2가지 loss함수에 대한 가중치값으로 우리가 결정해야 하는듯
        d_alpha = 0.5
        d_beta = 1 - d_alpha

        # Loss for real images
        real_pred, real_attr = discriminator(real_imgs)
        # d_real_loss = (adversarial_loss(real_pred, valid) + attribute_loss(real_attr, labels)) / 2
        d_real_loss = d_alpha * adversarial_loss(real_pred, valid) + d_beta * attribute_loss(real_attr, labels)
        # print('r',real_pred.shape)
        # print('valid', valid.shape)

        # Loss for fake images
        fake_pred, fake_attr = discriminator(out_final.detach())
        # d_fake_loss = (adversarial_loss(fake_pred, fake) + attribute_loss(fake_attr, fake_attr_gt)) / 2
        d_fake_loss = d_alpha * adversarial_loss(fake_pred, fake) + d_beta * attribute_loss(fake_attr, fake_attr_gt)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        # print(d_loss.type) # 원래(sgan)랑 type똑같음(<built-in method type of Tensor object at ...>). 둘다 float형태.

        # Calculate discriminator accuracy
        pred = np.concatenate([real_attr.data.cpu().numpy(), fake_attr.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_attr_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader_p), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(train_dataloader_p) + i
        if batches_done % opt.sample_interval == 0:
            save_image(out_final.data[:25], "finals/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(out_synth.data[:25], "synth/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(out_predictedM.data[:25], "masks/%d.png" % batches_done, nrow=5, normalize=True)
            torch.save(generator, "generator_paired%d.pt" % batches_done)
            torch.save(discriminator, "discriminator_paired%d.pt" % batches_done)

print("unpaired train")
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_dataloader_up):

        batch_size = opt.batch_size

        # Adversarial ground truths
        # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        valid = Variable(FloatTensor(batch_size, 1, 2, 2).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1, 2, 2).fill_(0.0), requires_grad=False)
        # fake_attr_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)
        fake_attr_gt = Variable(FloatTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # line 280, line 286 -> FloatTensor가 기대된다고 해서 LongTensor -> FloatTensor 로 바꿔봄 => 에러 안남

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))) -> 우리는 사용X

        # Generate a batch of images
        out_predictedM, out_InvertedM, out_synth, out_final = generator(real_imgs)  # discriminator와 loss 계산에 쓰이는 애들
        loss = sganloss([out_final,
                         out_predictedM,
                         out_InvertedM,
                         out_synth],
                         )
        # # Loss measures generator's ability to fool the discriminator

        validity, _ = discriminator(out_final)  # ?????????? : 해결했으니까 물음표 치우시길!
        # print('validity', validity.shape) # validity torch.Size([10, 1, 2, 2])
        # print('val', valid.shape) # val torch.Size([10, 1])

        g_loss = 0
        g_loss += w.lam4 * loss.smooth_loss()
        g_loss += w.lam5 * loss.l2_norm()
        g_loss += w.lam6 * adversarial_loss(validity, valid)

        print("loss:", g_loss)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # d_alpha, d_beta는 discriminator에 사용되는 2가지 loss함수에 대한 가중치값으로 우리가 결정해야 하는듯
        d_alpha = 0.5
        d_beta = 1 - d_alpha

        # Loss for real images
        real_pred, real_attr = discriminator(real_imgs)
        # d_real_loss = (adversarial_loss(real_pred, valid) + attribute_loss(real_attr, labels)) / 2
        d_real_loss = d_alpha * adversarial_loss(real_pred, valid) + d_beta * attribute_loss(real_attr, labels)
        # print('r',real_pred.shape)
        # print('valid', valid.shape)

        # Loss for fake images
        fake_pred, fake_attr = discriminator(out_final.detach())
        # d_fake_loss = (adversarial_loss(fake_pred, fake) + attribute_loss(fake_attr, fake_attr_gt)) / 2
        d_fake_loss = d_alpha * adversarial_loss(fake_pred, fake) + d_beta * attribute_loss(fake_attr, fake_attr_gt)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        # print(d_loss.type) # 원래(sgan)랑 type똑같음(<built-in method type of Tensor object at ...>). 둘다 float형태.

        # Calculate discriminator accuracy
        pred = np.concatenate([real_attr.data.cpu().numpy(), fake_attr.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_attr_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader_p), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(train_dataloader_p) + i
        if batches_done % opt.sample_interval == 0:
            save_image(out_final.data[:25], "finals/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(out_synth.data[:25], "synth/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(out_predictedM.data[:25], "masks/%d.png" % batches_done, nrow=5, normalize=True)
            torch.save(generator, "generator_unpaired%d.pt" % batches_done)
            torch.save(discriminator, "discriminator_unpaired%d.pt" % batches_done)
