import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

loss_CE_fn = nn.CrossEntropyLoss()
softmax_model = nn.Softmax(dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        return x

def model_eval(model, data_loader):
    # compute the validation accuracy
    count = 0
    right_count = 0
    for imgs, labels in data_loader:
        count += len(labels)
        with torch.no_grad():
            outs = softmax_model(model(imgs))
            outs = torch.argmax(outs, dim=1)
                
            # Update correct prediction numbers
            right_count += (outs == labels).sum().item()
    
    return right_count / count

def model_eval2(model, data_loader):
    # compute the validation loss
    count = 0
    loss = 0.0
    for imgs, labels in data_loader:
        count += len(labels)
        with torch.no_grad():
            outs = model(imgs)
            loss += loss_CE_fn(outs, labels)
    
    return loss / count

def OMD_solver(p, u, lr, alpha):
    n = len(p)
    log_p_new = np.log(p) + lr * u
    log_p_new_sorted = np.sort(log_p_new)
    p_new_sorted = np.exp(log_p_new_sorted)

    i_star = 0
    for i in range(n, 0, -1):
        if log_p_new_sorted[i-1] + np.log(1 - alpha * (i-1) / n) <= np.log(alpha/n) + np.log(p_new_sorted[i-1:].sum()):
            i_star = i+1
            break
    
    ss = p_new_sorted[i_star-1:].sum()
    ss_log = np.log(ss)
    p_new_argsort = np.argsort(log_p_new)
    p_hat = np.zeros(n)
    for i in range(n):
        if i+1 < i_star:
            p_hat[p_new_argsort[i]] = alpha / n
        else:
            p_hat[p_new_argsort[i]] = np.exp( log_p_new[p_new_argsort[i]] + np.log(1 - alpha * (i_star-1) / n) - ss_log )
    
    return p_hat / p_hat.sum()