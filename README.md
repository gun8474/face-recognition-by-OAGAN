# face recognition by occlusion aware gan


### TODO List
<img src="https://user-images.githubusercontent.com/47767202/117538385-e3223f00-b040-11eb-955c-bf317b293d16.png" width="70%">

- [ ] 데이터 수집 및 전처리(landmark검출->resize)
- [x] `지혜승건` -> Generator - Occlusion Aware Module
  - [x] 데이터 수집, 전처리되는대로 input image, demension바꾸기
  - [x] 아래 reference의 table1 참고하여 모델 구현
  - [x] 모델에서 predicted mask M 저장: 마지막 레이어의 (1x128x128 output) sigmoid 출력
  - [x] 모델에서 intermediate face feature(중간 피쳐) 저장
  - [x] mask M을 xocc(최초 input)과 중간피쳐 각각에 dot product
- [x] `소현나연` -> Generator - Face Completion Module
  - [x] input image: oa module(지혜승건팀)의 output중에서 intermediate face feature와 M의 dot product 결과
  - [x] 아래 reference의 table1 참고하여 모델 구현
  - [x] 모델의 출력을 inverted M과 dot product
  - [x] oa module(지혜승건팀)의 xocc와 M eltw 결과를 3의 바로위 결과와 sum(->discriminator의 input)
- [x] `소현나연` -> Discriminator
- [ ] `지혜승건` -> loss function(pair-perceptual, style, pixel, smooth, L2, adversarial / unpair-smooth, L2, adversarial)
- [ ] train
- [ ] ...



### reference
[💫Semi-supervised Natural Face De-Occlusion](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/9195444)  
[💫semi-supervised gan code - pytorch](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/sgan/sgan.py)  
[semi-supervised gan code - keras](https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py)  
[semi-supervised-learning gan code - tensorflow](https://github.com/nejlag/Semi-Supervised-Learning-GAN)  
[semi supervised gan code- keras](https://livebook.manning.com/book/gans-in-action/chapter-7/v-6/)  
[Semi Supervised GAN for image classification code - pytorch](https://nbviewer.jupyter.org/github/opetrova/SemiSupervisedPytorchGAN/blob/master/SemiSupervisedGAN.ipynb)  
[mask face dataset](https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset)
