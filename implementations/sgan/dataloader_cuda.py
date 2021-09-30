#!/usr/bin/env python
# coding: utf-8

# ## pytorch custom dataset loader


from PIL import Image
import os
from torchvision import transforms

'''
<dataset path>
dataset / paired_dataset / with_mask / 1,2,3,4,5,6,7,... (folder = class)
                         / without_mask / 1,2,3,4,5,6,7,...(folder = class)
dataset / unpaired_dataset / 1,2,3,4,5....

with_mask와 without_mask 사진은 match되어야 함!! (개수,사람,얼굴,각도 등 모든것)
'''


class OAGandataset():
    # folder_numbering : 사진이 folder별로 분류되어있는지 (일단 실험용으로 받은 데이터셋은 아님)

    def __init__(self, a=0, b=10000, paired=False, folder_numbering=False):
        self.paired = paired
        self.folder_numbering = folder_numbering
        self.img_size = 128

        if self.paired:
            self.dir_x = "./dataset/paired_dataset/with_mask"
            self.dir_y = "./dataset/paired_dataset/without_mask"

            folders = sorted(os.listdir(self.dir_y))[a:b+1]
            folders = sorted([f for f in folders if not f.startswith('.')])  # ignore .DS_store in macOS
            if self.folder_numbering:
                file_names = [os.listdir(os.path.join(self.dir_y, f)) for f in folders]
                self.file_name_y = sum(file_names, [])  # flatten
            else:
                self.file_name_y = folders
        else:
            self.dir_x = "./dataset/unpaired_dataset"
        self.label = []

        folders = sorted(os.listdir(self.dir_x))[a:b+1]
        folders = sorted([f for f in folders if not f.startswith('.')])  # ignore .DS_store in macOS
        # folders = sorted(folders)
        self.file_name = folders
        self.label = [n for n in range(a, b + 1)]
        # folder가 없는경우 불러온 file index가 label

    def __len__(self):  # folder 갯수 = 사람 수
        # print("train : 총 ", len(self.file_name), "장의 image")
        return len(self.file_name)

    def __getitem__(self, index):
        # print ("index :", index)
        trans = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                    transforms.ToTensor()])

        if self.paired:  # paired image인 경우
            dir = os.path.join(self.dir_x, self.file_name[index])
            dir_ = os.path.join(self.dir_y, self.file_name_y[index])

            if self.folder_numbering:
                dir = os.path.join(self.dir_x, self.label[index], self.file_name[index])
                dir_ = os.path.join(self.dir_y, self.label[index], self.file_name_y[index])

            img = Image.open(dir)
            x_occ = trans(img)

            label = self.label[index]

            img_ = Image.open(dir_)
            x_gt = trans(img_)

            return x_occ, x_gt, label

        else:  # pair가 없는 image인 경우
            dir = os.path.join(self.dir_x, self.file_name[index])
            if self.folder_numbering: # TODO: 데이터셋 정해졌으니 걍 folder numbering 필요없을거같은데 삭제 가능? -> 답변:
                dir = os.path.join(self.dir_x, self.label[index])
            img = Image.open(dir)
            x_occ = trans(img)

            label = self.label[index]

            return x_occ, label
