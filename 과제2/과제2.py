import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

init_w = 2.5
loss_thres = float('inf')
thres_step = 0
loss_arr = []
weight_arr = []
epsilon = 1e-6

def loss_function(w):
    return (1 / 50) * (w ** 4 + w ** 2 + 10 * w) + 0.5

def newton_optim(w_old):
    gradient = (1 / 50) * (4 * w_old ** 3 + 2 * w_old + 10)
    gradient_2 = (1 / 50) * (12 * w_old ** 2 + 2)
    
    w_new = w_old - (gradient / (gradient_2 + epsilon))
    
    return w_new

def animate(frame):
    global init_w, loss_thres, thres_step
    
    loss = loss_function(init_w)
    w_new = newton_optim(init_w)
    
    init_w = w_new
    
    # early stopping
    if loss < loss_thres:
        loss_thres = loss
        thres_step = 0
    else:
        thres_step += 1
        if thres_step == 10:
            print('Detect Global Minimum')
            print(f'loss : {loss}')
            anim.event_source.stop()  
    
    loss_arr.append(loss)
    weight_arr.append(w_new)
    
    print(f'step : {frame}, loss : {loss}, Weight : {w_new}')
    
    w = np.linspace(-5, 5, 1000)
    y = loss_function(w)
    
    plt.clf()
    plt.plot(w, y, label = 'Loss Fucntion')
    plt.plot(weight_arr, loss_arr, 'ro', markersize = 3,label = 'Loss over Weights')
    plt.legend()
    plt.title('Newton Method')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
fig = plt.figure(figsize=(6, 4))

anim = FuncAnimation(fig, animate, frames=1000, interval=100)
anim.save('Newton Optimzer.png', dpi = 80, writer='imagemagick')

plt.show()
