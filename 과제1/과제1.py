import numpy as np
from matplotlib import pyplot as plt

# ======================================
# steplength가 1일때 stationary point에 converge하는 속도가 나머지 steplength(0.1, 0.01)보다 더 빠르다
# steplength가 클수록 처음 initial value는 더 작았다.
# 0의 근처에 수렴하기는 하였지만, 0에 더 approximate하려면 더 작은 steplength를 필요로 하고 그에 따른 step이 증가해야한다.
# ======================================

def loss_function(w):
  return (1/50) * (w ** 4 + w ** 2 + 10 * w)

def gradient_descent(w_old, steplength):
  w_new = w_old - steplength * ((1/50) * (4 * w_old ** 3 + 2 * w_old + 10))
  return w_new

step_length = [1, 0.1, 0.01]
init_w = 2
num_step = 1000

weight_store = np.zeros((len(step_length), num_step))
loss_store = np.zeros((len(step_length), num_step))

for i, step_length in enumerate(step_length):
    w = init_w
    for j in range(num_step):
        w = gradient_descent(w, step_length)
        weight_store[i, j] = w
        loss_store[i, j] = loss_function(w)
        
loss_first = loss_store[0, :]
loss_second = loss_store[1, :]
loss_third = loss_store[2, :]

plt.figure(figsize = (6, 4))
plt.plot(loss_first, 'r', label = "LOSS_steplength = 1")
plt.plot(loss_second, 'g--', label = "LOSS_steplength = 0.1")
plt.plot(loss_third, 'b', label = "LOSS_steplength = 0.01")
plt.xlabel('Step')
plt.ylabel('LOSS')
plt.title('Gradient Descent')
plt.legend()
plt.savefig('gradient_descent_plot.png')  # 이미지를 저장합니다.
plt.show()