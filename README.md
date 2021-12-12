
## 1. 모델 환경

코드는 .py 파일로 작성되었으며, Spyder 혹은 Pycharm 등의 IDE로 실행 가능합니다. 모델의 학습을 진행한 하드웨어의 spec은 Intel(R) Core(TM) i7-8700 @3.20GHz CPU, Nvidia GeForce GTX 1080 Ti GPU이며 11GB RAM 입니다.

## 2. 모델 설명
### 1. 부정맥 탐지를 위한 SE-ResNet
본 경진대회에서 SPS LAB팀은 주어진 8-leads ECG 심전도 데이터 셋이 정상인지 부정맥인지를 높은 성능(ROC curve area = 0.97)으로 탐지할 수 있는 1-D convolution 기반의 SE-ResNet을 개발했습니다. 1-D convolution은 단방향 convolution 연산을 통해 고차원 시계열 데이터의 특징을 학습할 수 있으며, ResNet은 복잡한 딥러닝 모델 구조로 인한 기울기 소실 문제를 완화하기 위해 residual architectures를 도입한 모델입니다. ResNet은 residual 모듈을 통해 몇개의 레이어를 건너 뛰어 지속적으로 이전의 정보를 추가해줌으로써 학습시 기울기가 원활하게 전달되는 장점이 있습니다. 
본 경진대회에서 SPS LAB팀은 기존의 1-D convolution 기반의 ResNet 모델에 squeeze and excitation (SE) 모듈을 추가 적용하여 residual에 해당하는 feature map의 채널 별 영향도까지 고려하여 특징 추출이 가능한 SE-ResNet 모델을 개발했습니다. SE 모듈은 각 채널의 전역 공간 정보를 추출하는 average pooling, 모델의 복잡도를 제한하면서 일반화 성능을 높이기 위한 2개의 fully connected layer (FC), 그리고 residual의 채널별 영향도가 반영된 특징을 추출하기 위한 sigmoid 함수로 이루어져 있습니다. SE 모듈의 최종 출력 결과물은 기존의 residual에 선형 결합되어 요약이 되며, 이후 FC와 Sigmoid 활성화 함수를 통과하여 최종적으로 인풋으로 들어온 8-leads ECG 심전도 데이터가 정상(0)인지 부정맥(1)인지 판별하게 됩니다. 

![image](https://user-images.githubusercontent.com/30248006/145712957-e39abc45-7157-4db4-9ed0-4fb7865098b2.png)

모델에 입력되는 인풋의 사이즈는 5,000개입니다. 주어진 데이터의 대부분은 5,000개의 포인트로 이루어져 있지만, 일부 데이터는 5,000개 미만의 포인트로 이루어져 있습니다. 데이터가 4,950개 이상 5,000개 미만의 포인트인 경우, zero-padding을 적용하여 사이즈를 5,000개 포인트로 맞추었으며, 데이터가 4,950개 미만의 포인트인 경우, zero-padding을 적용하면 데이터가 왜곡될 가능성이 있기 때문에 학습에서 제외했습니다. 

![image](https://user-images.githubusercontent.com/30248006/145713090-23ba268f-959c-4f7a-9963-bea98525d521.png)

### 2. 주요 코드 설명
#### 1) BasicBlock Class
  * Residual connection에 squeeze and excitation (SE) 모듈을 추가 적용하여 Residual에 해당하는 feature map의 채널 별 영향도를 반영하기 위한 클래스
```Python
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, dilation=1):
        super(BasicBlock, self).__init__()
        ...
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #Squeeze and Excitation block
        SE = self.pooling(out)
        SE = SE.view(len(SE), self.planes)
        SE = self.fc1(SE)
        SE = self.fc2(SE)
        SE = self.sigmoid(SE)

        out = out * SE.view(len(SE), self.planes, 1)
        
        return out
```

#### 2) ResNet Class
  * 총 4번의 Residual connection layer와 FC, sigmoid 활성화 함수를 통과하여 최종적으로 인풋으로 들어온 심전도 데이터가 정상(0)인지 부정맥(1)인지 판별하는 클래스
```Python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, groups=1):
        super(ResNet, self).__init__()
        ...
    def _make_layer(self, block, planes, blocks, stride=1):
        ...
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adapavgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
```
#### 3) 모델 초기화
  * model : SE-ResNet 모델을 초기화
  * Loss : 정상과 부정맥을 판별하기 위해 binary cross entropy loss 사용
  * opt : Optimizer로 adaptive moment estimation (ADAM) 사용
  * Lr_scheduler : learning rate scheduler로, 학습이 진행되지 않을 경우 learning rate를 줄이는데 사용
  * path2weights : 학습 epoch 당 모델의 weight을 저장하기 위한 경로
  * path3weights : 모델의 테스트를 진행할 때 모델의 weight을 불러오기 위한 경로  
  * epoch : 학습 epoch 횟수
```Python
model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)
loss_func = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr = 0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.1, patience = 10)
path2weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 저장 위치
path3weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #최적 모델 불러오는 위치
epoch = 100
```

#### 4) 모델 학습과 검증
  * train_val : epoch당 모델의 학습과 검증을 하기 위한 함수
  * test : test 데이터로 모델을 평가하기 위한 함수
```Python
def train_val(model, params):
  ...
def test(model, params):
  ...
```

## 3. 성능 평가 검증을 위한 실행 가이드
### 1. 데이터 전처리
#### 1) 전처리_1.py 실행
전처리는 두가지 단계를 거쳐서 진행합니다. 첫번째 전처리 단계는 전처리_1.py 파일을 실행함으로써 가능하며 각각의 xml 파일을 npy파일로 변경해주는 작업입니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다. 
```Python
import os
import base64
import xmltodict
import array
import numpy as np
```
우선 xml 형태의 원본 데이터가 저장된 경로를 지정합니다. train_abnormal_dir, train_normal_dir, val_abnormal_dir, val_normal_dir은 각각 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터가 있는 폴더 경로입니다. 그리고 npy 파일로 바뀌어 저장이 가능한 폴더를 생성하여 경로(train_abnormal_dir_save,  train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save)를 지정합니다. 테스트를 수행할 경우 학습과 검증 데이터 경로 대신, 부정맥 테스트 데이터와 정상 테스트 데이터가 포함된 경로를 각각 설정해주어 전처리를 진행하면 됩니다. 학습시 6으로 시작하는 폴더 내의 데이터는 12-leads ECG 데이터 셋이지만, 5와 8로 시작되는 폴더와 데이터 형태를 맞춰주기 위해 8-leads 데이터 셋만 사용했습니다. 

```Python
#raw data
train_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/arrhythmia/'
train_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/normal/'
val_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/arrhythmia/'
val_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/normal/'
raw_data_dir_list = [train_abnormal_dir, train_normal_dir, val_abnormal_dir, val_normal_dir]
#save
train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
```
#### 2) 전처리_2.py 실행
두번째 전처리 단계는 전처리_2.py 파일을 실행함으로써 가능하며 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터들을 하나의 npy파일로 통합해주는 단계입니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다. 테스트를 수행할 경우 학습과 검증 데이터 경로 대신 부정맥 테스트 데이터와 정상 테스트 데이터가 포함된 경로를 각각 설정해주어 전처리를 진행하면 됩니다.

```Python
import os
import numpy as np
```
우선 전처리 1단계에서 만든 폴더 경로를 train_abnormal_dir_save, train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save 변수에 설정해줍니다. 이후 두번째 전처리 파일이 만들어질 폴더를 지정합니다. 코드 상에서는 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터를 위해 총 4번 설정을 해야 합니다.
```Python
train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
...
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_abnormal.npy', train_abnormal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_normal.npy', train_normal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_abnormal.npy', val_abnormal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_normal.npy', val_normal_data_x)
```

### 2. 모델 테스트
#### 1) HDAI_부정맥_진단_SPSLAB.py 실행
HDAI_부정맥_진단_SPSLAB.py 파일을 실행하여 모델의 테스트를 진행할 수 있습니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다.
```Python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
```
모델 학습과 검증시에는 전처리 2단계에서 진행했던 파일이 있는 경로 (train_abnormal_dir_save, train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save)를 지정해주어야 합니다. 테스트를 수행할 경우에는 부정맥 데이터와 정상 데이터의 경로 (test_abnormal_dir_save, test_normal_dir_save)만 지정해주어도 됩니다. 또한 모델 테스트를 진행 할 경우, 꼭 test_mode = True로 변경해주어야 합니다.

```Python
#train data 디렉토리
train_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_abnormal.npy'
train_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_normal.npy'
#validation data 디렉토리
val_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_abnormal.npy'
val_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_normal.npy'
# test data 디렉토리
test_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_abnormal.npy'
test_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_normal.npy'

test_mode = True
```
테스트가 끝난 이후 AUC ROC curve의 area를 계산한 plot을 생성하기 위해서 HDAI_부정맥_진단_SPSLAB.py 파일이 있는 경로에 plots 폴더를 생성해줍니다.

![image](https://user-images.githubusercontent.com/30248006/145714495-a8e68d42-1d5b-4949-9c10-9aec5c395d93.png)

모델 학습 시 모델의 가중치를 저장하기 위해 path2weights의 경로를 설정해주어야 하며, 모델 테스트의 경우에는 모델의 가중치를 불러오기 위해 path3weights의 경로만 설정해주어도 됩니다. 이때 .pt의 이름은 바뀌지 않아도 되지만 'model' 이름의 폴더가 있는 경로를 바꾸어주어야 합니다.
```Python
path2weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 저장 위치
path3weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 불러오는 위치
```






