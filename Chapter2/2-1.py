import numpy as np

init_w = np.array([2, 2])

# 목적함수
def objective_function(w):
    return np.tanh(4*w[0] + 4*w[1]) + np.max([0.4 * w[0] ** 2, 1]) + 1

# 방향벡터 (norm = 1)
def normalize_vector():
    weight = np.random.rand(2)
    weight /= np.linalg.norm(weight)
    return weight

# 1. 8번의 weight update
# 2. 1번간 1000번의 random direction을 알아야함
# 3. 1000번 중에서 가장 objective_function이 작은 값을 선택해야함
g = objective_function(init_w)
print(f'Init Objective Function : {g}')
for i in range(8):
    candidate_weight = []
    candidate_objective_function = []
    for _ in range(1000):
        weight = normalize_vector()
        new_g = objective_function(weight)
        if g > new_g:
            candidate_objective_function.append(new_g)
            candidate_weight.append(weight)
    
    if not candidate_objective_function or not candidate_weight:
        break
    
    minimize_objective_function_index = np.argmin(candidate_objective_function)
    g = candidate_objective_function[minimize_objective_function_index]
    
    print(f'Update Objective Function {i + 1} iterations {g}')