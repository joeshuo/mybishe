# -*- coding: utf-8 -*-
# @Time    : 2024/2/15 
# @Author  : sjx_alo！！
# @FileName: main.py
# @Algorithm ：
# @Description:



import torch
import numpy as np
import torch.nn as nn
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix, classification_report
import pywt
from scipy.io import loadmat


# 设置使用cpu 还是 gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 构建左边的模型
class bCNN(nn.Module):
    def __init__(self, channels):
        super(bCNN, self).__init__()
        # 模块的定义
        self.cnn = nn.Conv2d(channels, 1, kernel_size=(5,15), stride=3)
        self.maxpool = nn.MaxPool2d(5, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 输入经过卷积层
        output = self.cnn(x)
        # 输入经过最大池化
        output = self.maxpool(output)
        # 输入展平
        output = self.flatten(output)
        return output


# 1D-AX extractor
def linear_regression(data):
    num_trials, num_segments, num_samples, num_channels = data.shape
    x = np.linspace(1, num_samples, num_samples)

    K = np.zeros(shape=(num_trials, num_segments, num_channels))
    A = np.zeros(shape=(num_trials, num_segments, num_channels))
    for trial_idx, trial in enumerate(data):
        for segment_idx, segment in enumerate(trial):
            for channel_idx in range(num_channels):
                (c1, c0) = np.polyfit(x, segment[:, channel_idx], deg=1)
                t_mean = np.mean(segment[:, channel_idx])
                K[trial_idx, segment_idx, channel_idx] = c1
                A[trial_idx, segment_idx, channel_idx] = c1 * t_mean + c0

    return K, A

# 对数据分段扩充
def segment_data(data, seg_length):
    num_trials,num_channels, num_samples  = data.shape
    assert num_samples % seg_length == 0
    num_segments = num_samples//seg_length
    data = np.transpose(data, axes=(0,2,1))
    seg_eeg_data = np.empty(shape=(num_trials, num_segments, seg_length, num_channels ))

    for trial_idx in range(num_trials):
        for segment_idx in range(num_segments):
            lower_limit = segment_idx * seg_length
            upper_limit = lower_limit + seg_length
            seg_eeg_data[trial_idx, segment_idx] = data[trial_idx, lower_limit:upper_limit, :]

    return seg_eeg_data



# 将标签转换为 onehot编码
def map_to_one_hot(target):
    new_target = np.copy(target)
    unique = np.unique(target)
    new_labels = np.linspace(0, len(unique)-1, len(unique))
    mapping = dict(zip(unique, new_labels))
    for idx, label in enumerate(target):
        new_target[idx] = mapping[label]
    return F.one_hot(torch.LongTensor(new_target), num_classes=len(unique))

# LSTM结构
class bLSTM(nn.Module):
    def __init__(self, num_channels, num_reduced_channels, LSTM_cells):
        super(bLSTM, self).__init__()

        self.dense = nn.Linear(num_channels, num_reduced_channels)
        self.batch_norm = nn.BatchNorm1d(num_reduced_channels)
        self.lstm = nn.LSTM(num_reduced_channels, LSTM_cells, dropout=0.6, batch_first=True)
        self.fc = nn.Linear(LSTM_cells, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense(x)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.fc(x)
        x = self.softmax(x)
        return x

# 总的模型结构
class AlModel(nn.Module):
    def __init__(self, cnnChannels, num_channels, num_reduced_channels, LSTM_cells):
        super(AlModel, self).__init__()
        self.cnnModel = bCNN(cnnChannels)
        self.lstmModel = bLSTM(num_channels, num_reduced_channels, LSTM_cells)

        self.outLayer = nn.Linear(125, 4)

    def forward(self, cnnx, lstmx):

        out1 = self.cnnModel(cnnx)
        out2 = self.lstmModel(lstmx)
        out = torch.cat((out1, out2), dim=-1)
        out = self.outLayer(out)
        return out


def get_confusion_matrix(trues, preds):
    labels = []
    for i in range(len(set(labels))):
        labels.append(i)
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix
# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = []
    for i in range(len(set(labels))):
        labels.append(i)

    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix'+str(numer_ind+1)+'.png')
    plt.show()




# 获取数据
data_path = 'F:/DataSet/bci_dataset/BCICIV_2a_gdf2/'

# 对各个被试进行循环计算
for numer_ind in range(9):

    # 读取训练数据
    train_path = data_path + 'A0' + str(numer_ind+1) + 'T.gdf'
    raw = mne.io.read_raw_gdf(train_path)  # mne读取gdf数据
    events, _ = mne.events_from_annotations(raw)
    raw.load_data()
    raw.filter(2., 40., fir_design='firwin')  # 使用iir滤波器进行滤波
    raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude='bads')
    tmin, tmax = 1.003, 3.9997
    event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
    if numer_ind == 3:
        event_id = dict({'769': 5, '770': 6, '771': 7, '772': 8})
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
        train_label_path = 'F:/DataSet/bci_dataset/true_labels/' + 'A0' + str(numer_ind + 1) + 'T.mat'
        train_labels = np.squeeze(loadmat(test_label_path)['classlabel']-1, axis=-1)

    else:
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

        train_labels = epochs.events[:, -1] - 7
    train_data = epochs.get_data()

    # 读取测试数据
    test_path = data_path + 'A0' + str(numer_ind + 1) + 'E.gdf'
    test_label_path = 'F:/DataSet/bci_dataset/true_labels/'+ 'A0' + str(numer_ind + 1) + 'E.mat'
    true_labels = loadmat(test_label_path)['classlabel']

    raw = mne.io.read_raw_gdf(test_path)  # mne读取gdf数据
    events, _ = mne.events_from_annotations(raw)
    raw.load_data()
    raw.filter(2., 40., fir_design='firwin')  # 使用iir滤波器进行滤波
    raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude='bads')
    tmin, tmax = 1.003, 3.9997
    event_id = dict({'769': 7})
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    test_labels = np.squeeze(true_labels-1, axis=-1)
    test_data = epochs.get_data()


    # 实例化标准化
    scaler = StandardScaler()
    scaler1 = StandardScaler()

    # 对训练数据进行标准化操作  因为这里是三维数据 所以需要对其先进行转换变为二维数据 后标准化完成后 又重新变成三维数据
    train_data = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1)) \
        .reshape(train_data.shape[0], train_data.shape[1], -1)

    test_data = scaler1.fit_transform(test_data.reshape(test_data.shape[0], -1)) \
        .reshape(test_data.shape[0], test_data.shape[1], -1)


    # 左边的部分
    # 进行小波变换

    # 采样频率
    sampling_rate = 250
    # 小波函数
    # wavename = "cgau8"
    wavename = "cmor"
    # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    totalscal = 94
    # 计算小波函数的中心频率
    fc = pywt.central_frequency(wavename)
    # 常数c
    cparam = 2 * fc * totalscal
    # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    scales = cparam / np.arange(totalscal, 1, -1)
    scales = np.arange(1, 94)
    # 连续小波变换模块    cwtdata_xtrain得到小波变换的数据   得到训练数据
    [cwtdata_xtrain, frequencies] = pywt.cwt(train_data, scales, wavename, 1.0 / sampling_rate)
    # 对训练数据进行维度变换
    X_train_cwt = np.transpose(cwtdata_xtrain, axes=(1, 0, 2, 3))

    # 得到测试数据的小波变换
    [cwtdata_xtest, frequencies] = pywt.cwt(test_data, scales, wavename, 1.0 / sampling_rate)
    # 对测试数据进行维度变换
    X_test_cwt = np.transpose(cwtdata_xtest, axes=(1, 0, 2, 3))

    # 对LSTM模型所使用的的数据进行处理
    new_train_target = map_to_one_hot(train_labels)
    segmented_train_data = segment_data(train_data, 125)
    A_train, K_train = linear_regression(segmented_train_data)

    new_test_target = map_to_one_hot(test_labels)
    segmented_test_data = segment_data(test_data, 125)
    A_test, K_test = linear_regression(segmented_test_data)


    # 网络模型参数
    cnnChannels = X_train_cwt.shape[1]
    LSTM_cells = 8
    num_channels = 22
    num_reduced_channels = 6
    # 设置训练次数
    epochs = 20

    # 定义模型
    model = AlModel(cnnChannels, num_channels, num_reduced_channels, LSTM_cells).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters())
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()



    # 将训练数据打包
    train_data = Data.TensorDataset(torch.FloatTensor(X_train_cwt),
                                    torch.FloatTensor(A_train),
                               torch.LongTensor(train_labels))
    test_data = Data.TensorDataset(torch.FloatTensor(X_test_cwt),
                                    torch.FloatTensor(A_test),
                               torch.LongTensor(test_labels))

    train_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = Data.DataLoader(test_data , batch_size=64, shuffle=False)



    for epoch in range(epochs):
        model.train()
        tol_loss = 0.0
        train_preds = []
        train_trues = []

        for idx_batch, (cwtx, x, y) in enumerate(train_loader):
            # 将训练数据输入模型
            netout = model(cwtx.to(device), x.to(device))
            # 计算损失函数
            loss = loss_function(netout, y.to(device))
            # 固定格式  优化器梯度归零  损失反向传播 优化器更新
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()

            train_outputs = netout.argmax(dim=1)

            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(y.detach().cpu().numpy())
            # print(train_preds)
        sklearn_accuracy = accuracy_score(train_trues, train_preds)
        sklearn_precision = precision_score(train_trues, train_preds, average='micro')
        sklearn_recall = recall_score(train_trues, train_preds, average='micro')
        sklearn_f1 = f1_score(train_trues, train_preds, average='micro')

        print(
            "[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(epoch,
                                                                                                                     tol_loss,
                                                                                                                     sklearn_accuracy,
                                                                                                                     sklearn_precision,
                                                                                                                     sklearn_recall,
                                                                                                                     sklearn_f1))
        # 将训练过程结果保存到txt文件中  以追加的形式
        f = open('./res' + str(numer_ind+1) + '.txt', 'a+')
        f.write("[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(epoch,
                                                                                                                     tol_loss,
                                                                                                                     sklearn_accuracy,
                                                                                                                     sklearn_precision,
                                                                                                                     sklearn_recall,
                                                                                                                     sklearn_f1))
        f.write("\n")
        f.close()

    model.eval()
    test_pres = []
    test_trues = []
    for idx_batch, (test_cnnx, test_x, test_y) in enumerate(test_loader):
        test_netout = model(test_cnnx.to(device), test_x.to(device))
        # Comupte loss
        test_netout = test_netout.argmax(dim=1)
        test_pres.extend(test_netout.detach().cpu().numpy())
        test_trues.extend(test_y.detach().cpu().numpy())

    test_trues_np = np.array(test_trues).reshape(-1, 1)
    test_pres_np = np.array(test_pres).reshape(-1, 1)
    # 计算精度、precision、recall、f_score
    sklearn_accuracy = accuracy_score(test_trues_np, test_pres_np)
    sklearn_precision = precision_score(test_trues_np, test_pres_np, average='micro')
    sklearn_recall = recall_score(test_trues_np, test_pres_np, average='micro')
    sklearn_f1 = f1_score(test_trues_np, test_pres_np, average='micro')
    print(classification_report(test_trues_np, test_pres_np))
    conf_matrix = get_confusion_matrix(test_trues_np, test_pres_np)
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix)
    print("[sklearn_metrics] val_accuracy:{:.4f} val_precision:{:.4f} val_recall:{:.4f} val_f1:{:.4f}".format(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))

    f = open('./res' + str(numer_ind+1) + '.txt', 'a+')
    f.write("[sklearn_metrics] val_accuracy:{:.4f} val_precision:{:.4f} val_recall:{:.4f} val_f1:{:.4f}".format(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
    f.write("\n")
    f.close()
    #print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
    model.train()


