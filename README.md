
## 1. 모델 환경

.py 코드로 작성되었으며, pycharm



## 2. 모델 설명




## 3. 성능 평가 검증을 위한 실행 가이드
### 1. 전처리
 
```
#function to calculate loss and metric per epoch
def loss_epoch(epoch, model, loss_func, data_dl, label, sanity_check = False, opt = None, val = False):
    running_loss = 0.0
    running_metric_b = 0.0
    len_data = len(data_dl)

```

