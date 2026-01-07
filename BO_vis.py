import os
import torch
import numpy as np
import plotly.graph_objects as go
import warnings
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize

# 1. 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", message=".*qExpectedImprovement.*") 
warnings.filterwarnings("ignore", category=DeprecationWarning)

# [중요] 정밀도 설정 (Double Precision)
tkwargs = {"dtype": torch.double}

# 2. Objective Function 정의
def target_function(individuals):
    result = []
    for x in individuals:
        result.append(
            np.exp(-(x[0] - 2)**2)
            + np.exp(-(x[0] - 6)**2 / 10)
            + 1 / (x[0]**2 + 1))    
    return torch.tensor(result, **tkwargs)

# 3. 초기 데이터 생성 함수
def generate_initial_data(n):
    train_x = torch.rand(n, 1, **tkwargs) * 10  
    exact_obj = target_function(train_x).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

# 4. 시각화 함수 (Numpy 변환 적용됨)
def plot_gp(model, init_x, init_y, new_candidate=None, title="Bayesian Optimization Step"):
    test_x = torch.linspace(0, 10, 200, **tkwargs).unsqueeze(-1)
    
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean.squeeze()
        std = posterior.variance.sqrt().squeeze()
        lower = mean - 2 * std
        upper = mean + 2 * std

    # Tensor -> Numpy 변환
    test_x_np = test_x.squeeze().detach().cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()
    init_x_np = init_x.squeeze().detach().cpu().numpy()
    init_y_np = init_y.squeeze().detach().cpu().numpy()
    true_y_np = target_function(test_x).squeeze().detach().cpu().numpy()

    fig = go.Figure()

    # 1) 실제 함수
    fig.add_trace(go.Scatter(x=test_x_np, y=true_y_np, mode='lines', 
                             line=dict(color='black', dash='dash'), name='True Function'))
    # 2) 모델의 예측
    fig.add_trace(go.Scatter(x=test_x_np, y=mean_np, mode='lines', 
                             line=dict(color='blue'), name='GP Mean (AI 생각)'))
    # 3) 불확실성 영역
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_x_np, test_x_np[::-1]]), 
        y=np.concatenate([upper_np, lower_np[::-1]]),
        fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty (불확실성)'
    ))
    # 4) 관측 데이터
    fig.add_trace(go.Scatter(x=init_x_np, y=init_y_np, mode='markers',
                             marker=dict(color='red', size=8), name='Observed Data'))

    # 5) 새로운 후보 (있을 때만 그림)
    if new_candidate is not None:
        new_cand_np = new_candidate.squeeze().detach().cpu().numpy()
        if new_cand_np.ndim == 0: new_cand_np = np.array([new_cand_np])
        new_res_np = target_function(new_candidate).squeeze().detach().cpu().numpy()
        if new_res_np.ndim == 0: new_res_np = np.array([new_res_np])

        fig.add_trace(go.Scatter(x=new_cand_np, y=new_res_np,
                                 mode='markers', marker=dict(color='green', symbol='star', size=15),
                                 name='Next Candidate'))

    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    fig.show()


# 5. 다음 포인트 추천 및 모델 반환 함수 --> acquisition function
def get_next_points_and_model(init_x, init_y, best_init_y, bounds, n_points=1):
    single_model = SingleTaskGP(init_x, init_y, input_transform=Normalize(d=1, bounds=bounds))
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    
    EI = qExpectedImprovement(model=single_model, best_f=best_init_y)
    
    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates, single_model 


# --- 메인 실행 루프 ---

# 1. 초기 데이터 생성
n_initial_points = 5
init_x, init_y, best_init_y = generate_initial_data(n_initial_points) 
bounds = torch.tensor([[0.], [10.]], **tkwargs)

# -----------------------------------------------------------
# [추가된 부분] Step 0: 무작위 초기 데이터 시각화
# -----------------------------------------------------------
print("=== Initial State (Before Optimization) ===")
# 초기 데이터만으로 모델을 살짝 만들어서(fit) 그래프를 그립니다.
init_model = SingleTaskGP(init_x, init_y, input_transform=Normalize(d=1, bounds=bounds))
mll = ExactMarginalLogLikelihood(init_model.likelihood, init_model)
fit_gpytorch_mll(mll)

# new_candidate=None으로 설정하여 초록 별 없이 현재 상태만 봅니다.
plot_gp(init_model, init_x, init_y, new_candidate=None, title="Step 0: Initial Random Data")
# -----------------------------------------------------------


# 2. 최적화 루프 시작
n_runs = 10 
for i in range(n_runs):
    print(f"=== Optimization Run: {i+1}/{n_runs} ===")
    
    new_candidates, model = get_next_points_and_model(init_x, init_y, best_init_y, bounds, 1)
    
    # 시각화 (제목에 단계 표시)
    plot_gp(model, init_x, init_y, new_candidates, title=f"Step {i+1}: Optimization Run")
    
    new_results = target_function(new_candidates).unsqueeze(-1)
    print(f"New candidate: {new_candidates.item():.4f} -> Result: {new_results.item():.4f}")
    
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])
    
    best_init_y = init_y.max().item()
    print(f"Current Best: {best_init_y:.4f}\n")