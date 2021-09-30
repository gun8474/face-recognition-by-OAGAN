import torch
import torchvision.models as models
from torch import linalg as LA

cuda = True if torch.cuda.is_available() else False

class sganloss():
    # TODO: vgg_features를 loss함수마다 계산해야 해서 메모리를 엄청 잡아먹음. __init__에서 사진마다 feature 저장해주고, conv layer마다 받아오는게 아니라 pooling layer마다 feature 받아오게 수정
    def __init__(self, imgs, gt=None):
        self.final = imgs[0]
        self.M = imgs[1]
        self.inverse_M = imgs[2]
        # self.gt = imgs[3]
        self.synth = imgs[3]
        self.gt = gt

        self.vgg16 = models.vgg16(pretrained=True).cuda()
        self.vgg16_features = self.vgg16.features[:31]

        self.final_features = self.vgg16_features(self.final)
        self.gt_features = self.vgg16_features(self.gt)
        self.sy_features = self.vgg16_features(self.synth)

    def pixel_loss(self, alpha, beta):
        one = torch.matmul(self.inverse_M, (self.final - self.gt))
        norm_one = torch.norm(one, 1)

        two = torch.matmul(self.M, (self.final - self.gt))
        norm_two = torch.norm(two, 1)
        return alpha * norm_one + beta * norm_two


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
        a = self.sy_features
        b = self.gt_features
        c = self.final_features
        pc += torch.norm(a - b, 1)
        pc += torch.norm(c - b, 1)


        return pc


    def style_loss(self):
        sy = 0

        kn = 1 / (self.sy_features.size(1) * self.sy_features.size(2) * self.sy_features.size(3))

        s = self.sy_features
        sT = torch.transpose(s.detach(), 2, 3)

        g = self.gt_features
        gT = torch.transpose(g.detach(), 2, 3)

        f = self.final_features
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
            x_normalized += LA.norm(x[i])# norm default -> L2 :https://pytorch.org/docs/stable/linalg.html#torch.linalg.norm
        return x_normalized
