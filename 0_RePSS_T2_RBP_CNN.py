import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VVdataset(Dataset):
    def __init__(self, file_path,sheet_name):
        self.data = pd.read_excel(file_path,sheet_name=sheet_name)
        self.input_data = self.data.iloc[:, 0:888].values
        self.labels = self.data.iloc[:, -2:].values
        self.scaler = StandardScaler()
        self.scaler.fit(self.input_data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx, :]
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = self.scaler.transform(input_data.reshape(1, -1)).squeeze()
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        return input_data, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class CNN_Residual_Attention(nn.Module):
    def __init__(self, feature_size, out_channels, output_size):
        super(CNN_Residual_Attention, self).__init__()
        self.res_block1 = ResidualBlock(feature_size, out_channels[0])
        self.res_block2 = ResidualBlock(out_channels[0], out_channels[1])
        self.res_block3 = ResidualBlock(out_channels[1], out_channels[2])
        self.attention_local = nn.Linear(888, 888)  # 修改这里
        self.attention_global = nn.Linear(out_channels[2], out_channels[2])
        self.fc1 = nn.Linear(out_channels[2], 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        # Local Attention
        local_weights = F.softmax(self.attention_local(x), dim=2)
        x = x * local_weights
        # Global Attention
        global_weights = F.softmax(self.attention_global(x.mean(dim=2)), dim=1).unsqueeze(2)
        x = x * global_weights
        x = x.sum(dim=2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def bmc_loss(pred, target, noise_var):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var)
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

dataset = VVdataset('data/VV-small.xlsx',"train-age15to55")
train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
init_noise_sigma = 8.0
sigma_lr = 1e-2
model = CNN_Residual_Attention(feature_size=1, out_channels=[32,64,128], output_size=2)
criterion = BMCLoss(init_noise_sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})
model.to(device)
num_epochs=200
min_loss = 1000

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        # print(inputs.shape)
        inputs = inputs.unsqueeze(1)
        print(inputs.shape)
        inputs, labels = inputs.float().to(device), labels.float().to(device)  # 将输入和标签转换为 float 类型
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        # print(outputs)
        loss = criterion(outputs.squeeze(1), labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
    if epoch_loss<min_loss:
        min_loss=epoch_loss
        model_path = 'Res_Attention_BP.pth'
        print("saved")
        torch.save(model.state_dict(), model_path)