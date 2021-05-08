# face recognition by occlusion aware gan


### TODO List
<img src="https://user-images.githubusercontent.com/47767202/117538385-e3223f00-b040-11eb-955c-bf317b293d16.png" width="70%">

- [ ] 데이터 수집 및 전처리(landmark검출->resize)
- [ ] `지혜승건` -> Generator - Occlusion Aware Module
  - [ ] 데이터 수집, 전처리되는대로 input image, demension바꾸기
  - [ ] 아래 reference의 table1 참고하여 모델 구현
  - [ ] 모델에서 predicted mask M 저장: 마지막 레이어의 (1x128x128 output) sigmoid 출력
  - [ ] 모델에서 intermediate face feature(중간 피쳐) 저장
  - [ ] mask M을 xocc(최초 input)과 중간피쳐 각각에 dot product
- [ ] `소현나연` -> Generator - Face Completion Module
  - [ ] input image: oa module(지혜승건팀)의 output중에서 intermediate face feature와 M의 dot product 결과
  - [ ] 아래 reference의 table1 참고하여 모델 구현
  - [ ] 모델의 출력을 inverted M과 dot product
  - [ ] oa module(지혜승건팀)의 xocc와 M eltw 결과를 3의 바로위 결과와 sum(->discriminator의 input)
- [ ] Discriminator
- [ ] train
- [ ] ...



### reference
[Semi-supervised Natural Face De-Occlusion](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/9195444)
