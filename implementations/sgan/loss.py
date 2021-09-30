import numpy as np
# from sgan_main import *
import torch
# from gan_model import *
import torch.nn as nn
import torchvision.models as models
# import numpy.linalg
from torch import linalg as LA

class sganloss():
    # TODO: vgg_features를 loss함수마다 계산해야 해서 메모리를 엄청 잡아먹음. __init__에서 사진마다 feature 저장해주고, conv layer마다 받아오는게 아니라 pooling layer마다 feature 받아오게 수정
    def __init__(self, imgs , gt = None):
        self.final = imgs[0]
        self.M = imgs[1]
        self.inverse_M = imgs[2]
        self.synth = imgs[3]
        self.gt = gt

        # self.img = torch.randn([1, 3, 128, 128])
        self.vgg16 = models.vgg16(pretrained=True).cuda()
        # self.vgg16_features = [self.vgg16.features[:5],
        #                        self.vgg16.features[:10],
        #                        self.vgg16.features[:17],
        #                        self.vgg16.features[:24],
        #                        self.vgg16.features[:31]]
        self.vgg16_features = self.vgg16.features[:31]

        # self.M_features = [model(self.M) for model in self.vgg16_features]
        # self.iM_features = [model(self.inverse_M) for model in range(self.vgg16_features)]
        # self.final_features = [model(self.final) for model in self.vgg16_features]
        # self.final_features = [model(self.final) for model in self.vgg16_features]
        # self.gt_features = [model(self.gt) for model in self.vgg16_features]
        # self.sy_features = [model(self.synth) for model in self.vgg16_features]

    def pixel_loss(self, alpa, beta):
        one = torch.matmul(self.inverse_M, (self.final - self.gt))
        one = torch.norm(one, 1)
        one = alpa * one.detach()

        two = torch.matmul(self.M, (self.final - self.gt))
        two = torch.norm(two, 1)
        two = beta * two.detach()

        return one + two


    def smooth_loss(self):
        final = self.final
        M = self.M
        sm = 0
        for batch in range(len(final)):
            for channel in range(len(final[0])):
                for i in range(len(final[0][0])):
                    a = final[batch, channel, i, 1:]
                    b = final[batch, channel, i, :-1]
                    sm += torch.norm(a - b, 1)


                    c = final[batch, channel, 1:, i]
                    d = final[batch, channel, :-1, i]
                    sm += torch.norm(c - d, 1)

            for i in range(len(M[0][0])):
                a = final[batch, channel, i, 1:]
                b = final[batch, channel, i, :-1]
                sm += torch.norm(a - b, 1)

            c = final[batch, channel, 1:, i]
            d = final[batch, channel, :-1, i]
            sm += torch.norm(c - d, 1)

        return sm


    def perceptual_loss(self):
        pc = 0

        for i in range(len(self.vgg16_features)):
            a = self.sy_features[i]
            b = self.gt_features[i]
            c = self.final_features[i]
            pc += torch.norm(a - b, 1)
            pc += torch.norm(c - b, 1)


        return pc


    # print(vgg16_features[0](img).size(2))

    def style_loss(self):
        sy = 0

        for i in range(len(self.vgg16_features)):
            kn = 1 / (self.sy_features[i].size(1) * self.sy_features[i].size(2) * self.sy_features[i].size(3))

            s = self.sy_features[i]
            sT = torch.transpose(s.detach(), 2, 3)

            g = self.gt_features[i]
            gT = torch.transpose(g.detach(), 2, 3)

            f = self.final_features[i]
            fT = torch.transpose(f.detach(), 2, 3)

            a = torch.matmul(sT, s) - torch.matmul(gT, g)
            a = torch.norm(a, 1)
            b = torch.matmul(fT, f) - torch.matmul(gT, g)
            b = torch.norm(b, 1)
            sy += kn * (a + b)

        return sy

    def l2_norm(self):
        x_normalized = 0
        x = self.M.detach()
        for i in range(len(x)):
            # print(x[i].shape)
            x_normalized += LA.norm(x[i])
            # x_normalized+=np.linalg.norm(x[i], axis=1, ord=2)
        # print("dd")
        # x_normalized =sum(x_normalized[0])
        # print(x_normalized)
        # norm = x.norm(p=2, dim=1, keepdim=True)
        # x_normalized = x.div(norm.expand_as(x))
        return x_normalized

