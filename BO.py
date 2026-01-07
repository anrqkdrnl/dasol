import os
import torch
import numpy as np
import plotly
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", message=".*qExpectedImprovement.*") 
warnings.filterwarnings("ignore", category=DeprecationWarning)
tkwargs = {"dtype": torch.double}

#objective function 정의
def target_function(individuals):
    result = []

    for x in individuals:
        result.append(
            np.exp(-(x[0] - 2)**2)
            + np.exp(-(x[0] - 6)**2 / 10)
            + 1 / (x[0]**2 + 1))    
    return torch.tensor(result)

#objective funciton 시각화
import plotly.graph_objects as go

x = np.linspace(-2., 10., 100)
x_new = x.reshape((100,-1))
z = target_function(x_new)

data = go.Scatter(x=x, y=z, line_color='blue')

fig = go.Figure(data=data)
fig.update_layout(title='Target Function', xaxis_title='x', yaxis_title='output')
fig.show()

#초기 데이터 생성
##train_x = torch.rand(10,1, **tkwargs)*10 #10개의 랜덤 HF값을 볼 x좌표 생성 - [0,10]사이
##exact_obj = target_function(train_x).unsqueeze(-1) #10개의 랜덤 x에 대한 HF의 값 & unsqeeze : 차원추가 - (10) -> (10,1)
##best_obseved_value = exact_obj.max().item() #가장 큰 HF값을 추출
def generate_initial_data(n):
    train_x = torch.rand(n, 1, **tkwargs)*10  # 0~10 사이의 랜덤한 x 좌표 n개 생성
    exact_obj = target_function(train_x).unsqueeze(-1) #n개의 랜덤 x에 대한 HF의 값 
    best_obseved_value = exact_obj.max().item() #가장 큰 HF값을 추출
    return train_x, exact_obj, best_obseved_value #
generate_initial_data(20)
init_x, init_y, best_init_y = generate_initial_data(20) #함수 반환 값을 각각 변수에 할당
bounds = torch.tensor([[0.], [10.]], **tkwargs) #함수의 정의역을 0~10으로 설정

#모델의 정의 및 학습(fitting) - 준비
from botorch.models import SingleTaskGP, ModelListGP #GP모델을 라이브러리에서 불러옴
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood #GP모델 데이터를 보고 평가하는 지표를 불러옴
#모델 정의 - GP모델 생성 입력(init_x)과 출력(init_y)으로 초기 데이터 사용
single_model = SingleTaskGP(init_x, init_y)
#mll - Marginal Log Likelihood : 모델을 학습시키기 위한 손실 함수
mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
#Fit the model
from botorch import fit_gpytorch_mll
fit_gpytorch_mll(mll)

#acquisition function 정의
from botorch.acquisition.monte_carlo import qExpectedImprovement

EI = qExpectedImprovement(model=single_model, best_f=best_init_y) #획득 힘수를 EI로 정의

#획득 함수를 최적화 후 다음 후보점 도출
from botorch.optim import optimize_acqf
##candidates, _ = optimize_acqf( # optimize_acqf(획득함수 최적화 함수) : 찾아낸 x좌표, 그떄의 획득함수값 (2개)반환 - 획득함수값이 저장될 변수는 이름짓기 귀찮으므로 _로 처리
    ##acq_function=EI,#획득함수의 모델 정의
    ##bounds=bounds, #탐색할 범위 지정
    ##q=1,#한번에 찾을 후보점의 개수 - 3. 200개중 가장 값이 좋은 1개 선택하여 반환
    ##num_restarts=200, # 2. 512개중 가장 값이 좋은 200개를 선택
    ##raw_samples=512, # 1. 무작위로 512의 샘플을 뽑아 최적화 시작점으로 사용
    ##options={"batch_limit": 5, "maxiter": 200} #한번에 5개씩 계산, 답을 못찾아도 200번 반복했으면 멈춤
##)
##candidates

#반복 수행을 위한 함수 (모델생성 > 학습 > 다음 후보점 도출)을 하나의 함수로 설정한 것
from botorch.models.transforms.input import Normalize 
def get_next_points(init_x, init_y, best_init_y, bounds, n_points=1):
    single_model = SingleTaskGP(init_x, init_y, input_transform=Normalize(d=1, bounds=bounds)) # GP 모델은 0-1사이 가장 잘 작동하므로 0-10사이인 랜덤 값을 0-1사이로 변환; d=1 : 1차원, bounds=bounds : 변환 전 범위 지정 어디가 0이고 어디가 1로 될것인지
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    EI = qExpectedImprovement(model=single_model, best_f=best_init_y)
    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=2,
        raw_samples=5,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates

#Update the dataset
get_next_points(init_x, init_y, best_init_y, bounds, n_points=2)

#Full Bayesian Optimization loop
n_runs = 1 #20번 반복 수행
init_x, init_y, best_init_y = generate_initial_data(20) #초기 데이터 20개 생성
bounds = torch.tensor([[0.], [10.]]) #탐색할 범위 0~10 설정

for i in range(n_runs): # range - 0~19까지 숫자 행렬로 [0, 1, 2, ... 19] 생성(20개), in - ~안에 있는 숫자를 하나씩 꺼내서 i에 할당
    print(f"Nr. of optimization run: {i}")
    new_candidates = get_next_points(init_x, init_y, best_init_y, bounds, 1) #다음 후보점 1개 도출
    new_results = target_function(new_candidates).unsqueeze(-1) #도출된 후보점에 대한 HF값 계산

    print(f"New candidates are: {new_candidates}")
    #데이터셋 업데이트
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])
    #현재까지 찾은 최댓값 갱신
    best_init_y = init_y.max().item()
    print(f"Best point performs this way : {best_init_y}")
