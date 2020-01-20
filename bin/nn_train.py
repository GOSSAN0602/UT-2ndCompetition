import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
NFOLDS = 7
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=71)
splits = kf.split(train_x, train_y)

# epoch config
n_epochs = 1500
interval = 1

# for pred
y_preds = np.zeros([NFOLDS, test_x.shape[0]])
y_oof = np.zeros(train_x.shape[0])
tr_y_preds = np.zeros(train_x.shape[0])
tr_score = 0.0
va_score = 0.0

# make NN
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features*2)
        self.fc2 = nn.Linear(n_features*2, n_features*2)
        self.fc3 = nn.Linear(n_features*2, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# train NN
for fold_n, (tr_idx, va_idx) in enumerate(splits):
    # get batch data
    tr_x, va_x = Variable(torch.from_numpy(train_x[tr_idx]).float(),requires_grad=True), Variable(torch.from_numpy(train_x[va_idx]).float(),requires_grad=True)
    tr_y, va_y = Variable(torch.from_numpy(train_y[tr_idx]).float()), Variable(torch.from_numpy(train_y[va_idx]).float())

    net = Net(tr_x.shape[1])
    #optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_tr = np.zeros(int(n_epochs/interval))
    loss_va = np.zeros(int(n_epochs/interval))

    # epoch
    for i in range(n_epochs):
        net.train()
        optimizer.zero_grad()
        output = net(tr_x)
        loss = criterion(output, tr_y)
        loss.backward()
        optimizer.step()
        if (i+1) % interval ==0:
            net.eval()
            loss_tr[int(i/interval)] = criterion(tr_y, net(tr_x)).item()
            loss_va[int(i/interval)] = criterion(va_y, net(va_x)).item()
            if loss_va[int(i/interval)] <= loss_va[:(1+int(i/interval))].min():
                print(f'epoch: {i+1}  score improved  {loss_va[int(i/interval)]}')
                y_oof[va_idx] = mm.inverse_transform(net(va_x).detach().numpy()).reshape(-1,)
                y_preds[fold_n] = mm.inverse_transform(net(Variable(torch.from_numpy(test_x).float())).detach().numpy()).reshape(-1,)

    e_times = np.arange(interval, n_epochs+interval, interval)
    fig = plt.figure(figsize=(16, 9))
    plt.plot(e_times, loss_tr, label=f'{fold_n}_train')
    plt.plot(e_times, loss_va, label=f'{fold_n}_valid')

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

print(f"Scaled MSE(oof): {mean_squared_error(mm.inverse_transform(train_y), y_oof)}")
plt.legend()
plt.title(f"Scaled MSE(oof): {mean_squared_error(mm.inverse_transform(train_y), y_oof)}")
fig.savefig(f"./loss.png")

# make submission
import pdb;pdb.set_trace()
sub['quality'] = y_preds.mean(axis=0)
sub.to_csv("./submission.csv", index=False)
