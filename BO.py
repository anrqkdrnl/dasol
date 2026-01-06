import os
import torch
import numpy as np
import plotly

def target_function(individuals):
    result = []

    for x in individuals:
        result.append(
            np.exp(-(x[0] - 2)**2)
            + np.exp(-(x[0] - 6)**2 / 10)
            + 1 / (x[0]**2 + 1))    
    return torch.tensor(result)

import plotly.graph_objects as go

x = np.linspace(-2., 10., 100)
x_new = x.reshape((100,-1))
z = target_function(x_new)

data = go.Scatter(x=x, y=z, line_color='blue')

fig = go.Figure(data=data)
fig.update_layout(title='Target Function', xaxis_title='x', yaxis_title='output')
fig.show()
#  Generate initial data
train_x = torch.rand(10,1) #10개의 랜덤 HF값을 볼 x좌표 생성
exact_obj = target_function(train_x).unsqueeze(-1) #10개의 랜덤 x에 대한 HF의 값 & unsqeeze 항상 영역을 평평하게 하기 위해서
best_obseved_value = exact_obj.max().item() #가장 큰 HF값
def generate_initial_data(n=10):
    train_x = torch.rand(n, 1)  # Scale to [0, 10]
    exact_obj = target_function(train_x).unsqueeze(-1)
    best_obseved_value = exact_obj.max().item()
    return train_x, exact_obj, best_obseved_value
generate_initial_data(20)
init_x, init_y, best_init_y = generate_initial_data(20)
bounds = torch.tensor([[0.], [10.]])
#   Define the model
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
single_model = SingleTaskGP(init_x, init_y)
mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
#   Fit the model
from botorch import fit_gpytorch_mll
fit_gpytorch_mll(mll)
#   Define the acquisition function
from botorch.acquisition.monte_carlo import qExpectedImprovement

EI = qExpectedImprovement(model=single_model, best_f=best_init_y)
#   Optimize the acquisition function
from botorch.optim import optimize_acqf
candidates, _ = optimize_acqf(
    acq_function=EI,
    bounds=bounds,
    q=1,
    num_restarts=200,
    raw_samples=512,
    options={"batch_limit": 5, "maxiter": 200}
)

candidates
#   Evaluate the candidates using the objective function
def get_next_points(init_x, init_y, best_init_y, bounds, n_points=1):
    single_model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    EI = qExpectedImprovement(model=single_model, best_f=best_init_y)
    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=200,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates
#   Update the dataset
get_next_points(init_x, init_y, best_init_y, bounds, n_points=2)
#  Full Bayesian Optimization loop
n_runs = 10
init_x, init_y, best_init_y = generate_initial_data(20)
bounds = torch.tensor([[0.], [10.]])

for i in range(n_runs):
    print(f"Nr. of optimization run: {i}")
    new_candidates = get_next_points(init_x, init_y, best_init_y, bounds, 1)
    new_results = target_function(new_candidates).unsqueeze(-1)

    print(f"New candidates are: {new_candidates}")
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
    print(f"Best point performs this way : {best_init_y}")
