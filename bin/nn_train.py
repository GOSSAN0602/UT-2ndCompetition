import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import gc

# load data
#data_folder = '../input/pH_comp/'
data_folder = './input/pH_critic_comp/'
train = pd.read_csv(data_folder+'train.csv')
train_y = np.array(train.loc[:,['quality']])
train_x = np.array(train.drop(['quality','density'],axis=1))
test_x = pd.read_csv(data_folder+'test.csv')
test_x = np.array(test_x.drop('density',axis=1))
sub = pd.read_csv(data_folder+'submission.csv')

# preprocess for NN
mm = MinMaxScaler()
mm.fit(np.vstack([train_x,test_x]))
train_x = mm.transform(train_x)
test_x = mm.transform(test_x)
mm = MinMaxScaler()
mm.fit(train_y)
train_y = mm.transform(train_y)

# make fold
NFOLDS = 6
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=71)
splits = kf.split(train_x, train_y)

# epoch config
n_epochs = 2000
interval = 1
batch_size = 128

# for pred
y_preds = np.zeros([NFOLDS, test_x.shape[0]])
y_oof = np.zeros(train_x.shape[0])
y_oof_df = pd.DataFrame()
y_oof_df['quality'] = y_oof
tr_y_preds = np.zeros(train_x.shape[0])
tr_score = 0.0
va_score = 0.0

# make NN
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features*3)
        self.fc2 = nn.Linear(n_features*3, n_features*2)
        self.fc3 = nn.Linear(n_features*2, 1)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# train NN
fig = plt.figure(figsize=(16, 9))
for fold_n, (tr_idx, va_idx) in enumerate(splits):
    # get batch data
    tr_x, va_x = Variable(torch.from_numpy(train_x[tr_idx]).float(),requires_grad=True), Variable(torch.from_numpy(train_x[va_idx]).float(),requires_grad=True)
    tr_y, va_y = Variable(torch.from_numpy(train_y[tr_idx]).float()), Variable(torch.from_numpy(train_y[va_idx]).float())
    n_iter = int(tr_x.shape[0] / batch_size)+1
    batch_idx = np.arange(tr_x.shape[0])
    net = Net(tr_x.shape[1])
    #optimizer = optim.SGD(net.parameters(), lr=0.01)
    #optimizer = optim.AdamW(net.parameters(), lr=0.001)
    #optimizer =  optim.RAdam(net.parameters(), lr = 0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    criterion = nn.MSELoss()
    loss_tr = np.zeros(int(n_epochs/interval))
    loss_va = np.zeros(int(n_epochs/interval))

    # epoch
    for i in range(n_epochs):
        batch_idx = np.random.permutation(batch_idx)
        net.train()
        for iter in range(n_iter):
            optimizer.zero_grad()
            output = net(tr_x[batch_idx[iter*batch_size:(iter+1)*batch_size]])
            loss = criterion(output, tr_y[batch_idx[iter*batch_size:(iter+1)*batch_size]])
            loss.backward()
            optimizer.step()
        cos_lr_scheduler.step()
        if (i+1) % interval ==0:
            net.eval()
            loss_tr[int(i/interval)] = criterion(tr_y, net(tr_x)).item()
            loss_va[int(i/interval)] = criterion(va_y, net(va_x)).item()
            if loss_va[int(i/interval)] <= loss_va[:(1+int(i/interval))].min():
                print(f'epoch: {i+1}  score improved  {loss_va[int(i/interval)]}')
                y_oof[va_idx] = mm.inverse_transform(net(va_x).detach().numpy()).reshape(-1,)
                y_preds[fold_n] = mm.inverse_transform(net(Variable(torch.from_numpy(test_x).float())).detach().numpy()).reshape(-1,)

    e_times = np.arange(interval, n_epochs+interval, interval)
    #fig = plt.figure(figsize=(16, 9))
    plt.plot(e_times, loss_tr, label=f'{fold_n+1}_train',color='red')
    plt.plot(e_times, loss_va, label=f'{fold_n+1}_valid',color='blue')

    # evaluate model
    net.eval()
    #y_oof[va_idx] = mm.inverse_transform(net(va_x).detach().numpy()).reshape(-1,)
    tr_score += criterion(tr_y, net(tr_x)).item() / NFOLDS
    va_score += criterion(va_y, net(va_x)).item() / NFOLDS
    #y_preds += mm.inverse_transform(net(Variable(torch.from_numpy(test_x).float())).detach().numpy()).reshape(-1,) / NFOLDS

    print(f"Fold {fold_n + 1} | MSE(Train): {loss_tr[-1]}")
    print(f"Fold {fold_n + 1} | MSE(Valid): {loss_va[-1]}")

    del tr_x, tr_y, va_x, va_y
    gc.collect()

# make submission
oof_mse = mean_squared_error(mm.inverse_transform(train_y), y_oof)
os.mkdir(f'./sub/{oof_mse}')
sub['quality'] = y_preds.mean(axis=0)
sub.to_csv(f"./sub/{oof_mse}/submission.csv", index=False)
y_oof_df['quality'] = y_oof
y_oof_df.to_csv(f"./sub/{oof_mse}/y_oof.csv", index=False)

# save fig
print(f"Scaled MSE(oof): {oof_mse}")
plt.legend()
plt.title(f"Scaled MSE(oof): {oof_mse}")
fig.savefig(f"./sub/{oof_mse}/loss.png")
