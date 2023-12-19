import os
import time
import copy
import random

import numpy as np

import torch
import torch.optim as optim
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# seed 고정함수 정의
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 학습 과정에서 사용할 함수
# 현재 optimizer의 학습률을 반환함
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# 학습 과정에서 사용할 함수
# 각 배치별로 rss를 반환함
def rss_batch(output, target):
    rss = torch.sum(torch.pow(output - target, 2)).item()
    
    return rss


# 각 배치별로 loss를 계산하여 반환함
def loss_batch(loss_func, output, target, opt=None):
    # Argument로 받은 loss function은 위에서 정의한 교차엔트로피오차
    loss = loss_func(output, target)
    rss_b = rss_batch(output, target)

    # Optimizer는 위에서 Adam으로 지정함
    # 따로 설정하지 않았다면 최적화하지 않겠다는 뜻
    # 설정되어 있다면 최적화(역전파) 진행 -> 아래 학습 진행 코드에서 이 함수를 불러서 역전파까지 진행
    if opt is not None:
        opt.zero_grad()
        loss = torch.sqrt(loss) # RMSE 적용
        loss.backward()
        opt.step()

    # 계산된 loss, 정확도 반환
    return loss.item(), rss_b


# 각 Epoch마다 loss와 정확도를 계산하여 반환
def loss_epoch(model, loss_func, dataset_dl, DEVICE, sanity_check=False, opt=None):
    running_loss = 0.0 # 배치마다 나온 loss값을 누적해서 더할 변수 초기화
    running_metric = 0.0 # 배치마다 나온 정답개수를 누적해서 더할 변수 초기화
    
    len_data = dataset_dl.dataset.X.shape[0]#len(dataset_dl.dataset.X) # 총 데이터 개수를 계산하여 저장
    # dataset_dl은 train/test 데이터셋으로 정의한 dataloader -> 다시 데이터셋에 접근하기 위해서 loader.dataset 사용
    
    for xb, yb in dataset_dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        output = model(xb) # 모델의 예측값 저장

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b # 배치에서 나온 loss를 누적
        

        
        if metric_b is not None:
            running_metric += metric_b # 평가 지표 누적
        
        if sanity_check is True:
            # 작은 데이터를 돌려서 학습과정이 에러 없이 돌아가는 지 확인을 위함
            # 처음에는 True로 설정하고, 실제 학습 시에는 무조건 False
            break

    # 데이터 개수로 loss를 나눔
    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

# 학습 함수 정의
def train_val(model, params):
    # 학습 시작 전 값 초기화

    # params는 dictionary형 변수로 모델 학습에 필요한 여러 정보를 담고있음
    # 아래 쪽 cell에서 값 할당
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    DEVICE=params["DEVICE"]
    # EARLY STOPPING PARAMETERS
    early_stopping=params["early_stopping"]
    patience_limit=params["patience_limit"] # 몇 번의 epoch까지 지켜볼지를 결정

    patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록

    # loss, 정확도 변화를 추적할 변수
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memory error
    # copy 패키지는 변수를 복사하는 패키지
    # copy.copy는 원본 변수의 주소값만 참조하는 형식
    # copy.deepcopy는 해당 변수 값을 새로운 주소값에 할당하는 식
    # GPU에 있는 정보에 접근하려고 넣은듯
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf') # 제일 좋은 loss를 낸 모델을 판별하기 위함, 무한대로 초기화
    best_epoch = 0

    start_time = time.time() # 학습 소요시간을 에폭마다 출력하기위해 학습 시작 시간 저장

    # 학습 시작
    for epoch in range(num_epochs):
        current_lr = get_lr(opt) # 현재 학습률 확인
        

        model.train() # train 모드로 전환
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, DEVICE, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        if val_dl != None:
            model.eval() # eval 모드로 전환하여 validation셋에 대한 값 계산
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model, loss_func, val_dl, DEVICE, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            # validation loss가 가장 낮은 모델의 파라미터를 저장
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

                torch.save(model.state_dict(), path2weights)
                print('Copied best model weights!')
                print('Get best val_loss')

            # 학습률을 줄일지 여부 확인
            lr_scheduler.step(val_loss)
        else:
            val_loss = 0.0
            val_metric = 0.0

        # 하나의 에폭에 대해서 학습 정보 출력
        # print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        if epoch % 100 == 0:
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
            print('train loss: %.6f, val loss: %.6f, mse: %.2f, time: %.4f min' %(train_loss, val_loss, val_metric, (time.time()-start_time)/60))
            print('-'*10)

        # EARLY STOPPING
        if val_dl != None:
            if early_stopping == True:
                if val_loss > best_loss: # loss가 개선되지 않은 경우
                    patience_check += 1

                    if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                        print('EARLY STOPPED AT EPOCH--' + str(epoch))
                        break

                else: # loss가 개선된 경우
                    best_loss = val_loss
                    patience_check = 0

    if val_dl == None:
        torch.save(model.state_dict(), path2weights)
    
    # 학습 종료 후 학습 history 저장
    np.save(path2weights[:-3] + '_loss_history.npy', loss_history)
    np.save(path2weights[:-3] + '_metric_history.npy', metric_history)
    # 학습 종료 후에 Best 모델의 파라미터로 갈아 끼우기
    model.load_state_dict(best_model_wts)
    
    return model, loss_history, metric_history, best_epoch, best_loss

def timeseriesKFoldCV(objModel, objDataset, objDataloader,  params):
    X = params['X']
    Y = params['Y']
    
    tscv_train_idx = params['tscv_train_idx']
    tscv_valid_idx = params['tscv_valid_idx']
    numBatch = params['batch_size']
    
    LR = params['learning_rate']
    
    DEVICE = params["DEVICE"]

    seed = 42
    fold_losses = []

    num_folds = len(tscv_train_idx)
    for i, (train_idx, valid_idx) in enumerate(zip(tscv_train_idx, tscv_valid_idx)):
        train_ds = objDataset(X[train_idx], Y[train_idx])
        train_dl = objDataloader(train_ds, batch_size=numBatch, shuffle=True)
        valid_ds = objDataset(X[valid_idx], Y[valid_idx])
        valid_dl = objDataloader(valid_ds, batch_size=numBatch, shuffle=True)
        
        params['train_dl'] = train_dl
        params['val_dl'] = valid_dl

        print(f'Train on fold-{i}')
        seed_everything(seed)
        fold_model = objModel(6).to(DEVICE)
        opt = optim.Adam(fold_model.parameters(), lr = LR)
        params['optimizer'] = opt
        # fold_model = copy.deepcopy(model).to(DEVICE)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor = 0.5, patience = 250)
        params['lr_scheduler'] = lr_scheduler
        _, _, _, _, best_loss = train_val(fold_model, params)
        fold_losses.append(best_loss)

        del params['train_dl']
        del params['val_dl']
    fold_result = np.mean(fold_losses)
    
    print(f'Averaged loss for {num_folds}-fold: {fold_result}')
    return fold_result, params
    
    