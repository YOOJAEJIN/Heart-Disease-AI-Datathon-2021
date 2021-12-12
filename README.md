
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

#### 3) 데이터 불러오기
  * train_abnormal_dir_save: 전처리된 train 비정상 데이터 npy 파일 경로  
  * train_normal_dir_save: 전처리된 train 정상 데이터 npy 파일 경로
  * val_abnormal_dir_save: 전처리된 validation 비정상 데이터 npy 파일 경로
  * val_normal_dir_save: 전처리된 validation 정상 데이터 npy 파일 경로
  * test_abnormal_dir_save: 전처리된 test 비정상 데이터 npy 파일 경로
  * test_normal_dir_save: 전처리된 test 정상 데이터 npy 파일 경로
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
```

#### 4) 모델 초기화
  * model : SE-ResNet 모델을 초기화
  * Loss : 정상과 부정맥을 판별하기 위해 binary cross entropy loss 사용
  * opt : Optimizer로 adaptive moment estimation (ADAM) 사용
  * Lr_scheduler : learning rate scheduler로, 학습이 진행되지 않을 경우 learning rate를 줄이는데 사용
  * path2weights : 학습 epoch 당 모델의 weight을 저장하기 위한 경로
  * path2weights : 모델의 테스트를 진행할 때 모델의 weight을 불러오기 위한 경로  
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

#### 5) 모델 학습과 검증
  * train_val : epoch당 모델의 학습과 검증을 하기 위한 함수
  * test : test 데이터로 모델을 평가하기 위한 함수
```Python
def train_val(model, params):
  ...
def test(model, params):
  ...
```

## 3. 성능 평가 검증을 위한 실행 가이드
### 1. 전처리
 
```Python
#function to calculate loss and metric per epoch
def loss_epoch(epoch, model, loss_func, data_dl, label, sanity_check = False, opt = None, val = False):
    running_loss = 0.0
    running_metric_b = 0.0
    len_data = len(data_dl)

```

