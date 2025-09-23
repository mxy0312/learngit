import numpy as np
import matplotlib.pyplot as plt
x = [1.0,2.0,3.0]
y = [2.0,4.0,6.0]

def forwaed(x):
    return x * w

def loss(x,y):
    y_pred = forwaed(x)
    return (y_pred -y)*(y_pred -y)

w_list = []
mse_list = []
for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    for b in np.arange(0.0,4.1,0.1):
        print('b=',b)
    l_sum = 0
    for x_val,y_val in zip(x,y):
        y_pred_val = forwaed(x_val)
        loss_val = loss(x_val,y_val)
        l_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE=',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
