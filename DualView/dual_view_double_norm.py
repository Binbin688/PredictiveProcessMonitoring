# encoding=utf8

import math
import random
import re
import numpy as np
import csv
import gc
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pytorchtools
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)  # 指定GPU


def read_log(log_add, case_num_):
    log = [[] for i in range(case_num_)]
    with open(log_add, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            log[int(row['case']) - 1].append(row)
    f.close()
    return log


def get_att_values(log_add, atts):
    att_dicts = dict()
    for att in atts:
        att_dicts[att] = []
    with open(log_add, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            for att in atts:
                if str(row[att]) != 'nan':
                    att_dicts[att].append(row[att])
    f.close()
    new_att_dicts = dict()
    for att in atts:
        new_att_dicts[att] = list(set(att_dicts[att]))

    return new_att_dicts


def time_encoding(encoding_length, target_attribute, attribute_values):
    encoding_result = dict()
    for i in range(len(attribute_values[target_attribute])):
        encoder_t = [0 for i in range(encoding_length - len(str(attribute_values[target_attribute][i])))]
        for s in str(attribute_values[target_attribute][i]):
            encoder_t.append(int(s))
        if max(encoder_t) != 0:
            new_encoder_t = []
            for s in encoder_t:
                new_encoder_t.append(round(s / max(encoder_t), 4))
            encoding_result[attribute_values[target_attribute][i]] = new_encoder_t
        else:
            encoding_result[attribute_values[target_attribute][i]] = encoder_t

    return encoding_result


def one_hot_encoding(encoding_length, target_attribute, attribute_values):
    """
    对目标属性进行one-hot编码，并以字典形式返回编码结果，时间戳除外
    :param encoding_length: int, 编码长度
    :param attribute_values: dict, 所有属性值
    :param target_attribute: str, 目标属性名称
    :return:encoding_result, dict, 对应属性的每一个值的编码结果，以字典形式输出，例如：{'org:resource': 00010, ...}
    """
    encoding_result = dict()

    for i in range(len(attribute_values[target_attribute])):
        init_encoding = [0 for i in range(encoding_length)]  # 初始化编码
        init_encoding[- (i + 1)] = 1
        encoding_result[attribute_values[target_attribute][i]] = init_encoding

    return encoding_result


def trans_to_prob(labels_, num_label_):
    batch = len(labels_)
    label_probs_ = [[0 for i in range(num_label_)] for j in range(batch)]

    for m in range(len(labels_)):
        label_probs_[m][int(labels_[m])] = 1

    return label_probs_


def dataset_split(a_list, train_percent=0.8):
    train_num = math.ceil(len(a_list) * train_percent)  # 训练集数量

    whole_index = [i for i in range(len(a_list))]
    final_train_index = random.sample(range(0, len(a_list)), train_num)
    valid_index = list(set(whole_index) - set(final_train_index))

    final_train_log_ = [a_list[i] for i in final_train_index]
    valid_log_ = [a_list[i] for i in valid_index]

    return final_train_log_, valid_log_


class CsvEventLog:
    """
    用来处理xes文件事件日志
    """

    def __init__(self, eventlog_address, attributes, row_num, min_length=3, label_attribute=None):

        self.log = read_log(eventlog_address, row_num)  # 事件日志, [[dict,dict,...],[],...,[]]
        self.min_length = min_length  # 最短的trace长度, 包括下一个事件, 低于self.min_length的都会被忽略
        self.max_length = max([len(trace) for trace in self.log])  # 最长的trace长度, 用来确定事件级LSTM的输入维度
        self.attributes = attributes  # list, 所选属性
        self.label_attribute = label_attribute
        self.attribute_values = get_att_values(eventlog_address, self.attributes)  # dict, 所有属性值
        self.max_atts = max([len(att_list) for att_list in self.attribute_values.values()])
        self.encoding_length = self.max_atts if self.max_atts > 10 else 10  # 属性编码的长度
        self.attribute_encodings = dict()
        for attribute in self.attributes:
            if attribute not in ['Complete Timestamp', 'time:timestamp', 'ComplaintThemeID', 'dateFinished']:
                self.attribute_encodings[attribute] = one_hot_encoding(self.encoding_length, attribute,
                                                                       self.attribute_values)  # 其它属性编码结果
            else:
                self.attribute_encodings[attribute] = time_encoding(self.encoding_length, attribute,
                                                                    self.attribute_values)  # 时间属性编码结果
        self.label_encoding = dict()
        for i in range(len(self.attribute_values[label_attribute])):
            self.label_encoding[self.attribute_values[label_attribute][i]] = i  # 事件(concept:name)标记结果

    def trace_encoding(self, a_trace):
        """
        one-hot编码, 将XES中的trace转化为tensor格式的数据
        :param a_trace: list[dict], [dict,dict,...]
        :return: list格式的trace,可以直接输入进神经网络
        """
        trace_encoding = []
        for i in range(len(a_trace)):
            event_encoding = []
            for attribute in self.attributes:
                if a_trace[i][attribute] in self.attribute_values[attribute]:
                    event_encoding.append(self.attribute_encodings[attribute][a_trace[i][attribute]])
                else:
                    event_encoding.append([0 for i in range(self.encoding_length)])
            trace_encoding.append(event_encoding)

        return trace_encoding  # list[list]

    def fit_prefix(self, a_prefix):
        """
        给prefix补零，使长度统一
        :param a_prefix: list, 一个待补零的prefix
        :return: list, 已经补充完整的prefix
        """
        unfitted_prefix = []
        fit_list = [[0 for i in range(self.encoding_length)] for j in range(len(a_prefix[0]))]
        if len(a_prefix) < self.max_length - 1:
            for i in range(self.max_length - len(a_prefix) - 1):
                unfitted_prefix.append(fit_list)
        new_prefix = unfitted_prefix + a_prefix  # 前序补零

        return new_prefix

    def dataset_encoding(self, a_log):
        """
        对数据集进行标注
        :param a_log: log, 待处理的log
        :return: list, 分别表示数据集的input和label
        """
        print("-----------处理数据集-----------")
        x_att_encodings, y_att_encodings = [], []
        for trace in tqdm(a_log):
            if len(trace) >= self.min_length:
                encoding = self.trace_encoding(trace)  # [len(trace), attribute_num, encoding_length], list[list]
                for i in range(len(trace) - self.min_length + 1):
                    prefix = encoding[:(i + self.min_length - 1)]  # list[list], [2, 3, 398]
                    prefix = self.fit_prefix(prefix)
                    x_att_encodings.append(prefix)
                    label = self.label_encoding[trace[i + self.min_length - 1][self.label_attribute]]
                    y_att_encodings.append(label)

        return x_att_encodings, y_att_encodings


class ResCell(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super(ResCell, self).__init__()

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.stride = stride

        self.cnn1 = nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.mid_channel)
        self.cnn2 = nn.Conv2d(self.mid_channel, self.mid_channel, kernel_size=3, stride=self.stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.mid_channel)
        self.cnn3 = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.out_channel)

        self.cnn4 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=self.stride)
        self.bn4 = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):  # x: [32,3,73,59]

        y = F.relu(self.bn1(self.cnn1(x)))
        y = F.relu(self.bn2(self.cnn2(y)))
        y = self.bn3(self.cnn3(y))
        x = self.bn4(self.cnn4(x))
        out = F.relu(x + y)

        return out


class ResBlock(nn.Module):
    def __init__(self, att_channel, output_dim):
        super(ResBlock, self).__init__()

        self.att_channel = att_channel
        self.output_dim = output_dim
        self.res_net1 = ResCell(self.att_channel, 64, 64, stride=2)
        self.res_net2 = ResCell(64, 128, 128, stride=2)
        self.res_net3 = ResCell(128, 256, 256, stride=2)
        self.res_net4 = ResCell(256, 512, 512, stride=2)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, self.output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        y = self.res_net1(x)  # 32,64,7,11
        y = self.res_net2(y)  # 32, 128, 4, 6
        y = self.res_net3(y)  # 32, 256, 2, 3
        y = self.res_net4(y)  # 32, 512, 1, 2
        out = torch.squeeze(self.pooling(y))  # 32, 512
        # out = self.dropout(out)
        out = self.linear(out)
        # out = F.softmax(out, dim=1)

        return out


class SelfAttention(nn.Module):
    def __init__(self, batch_size, max_prefix_length, hidden_dim, output_dim,
                 encoding_length):  # batch_size=32, max_prefix_length=21, hidden_dim=50, label_num=3
        super(SelfAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoding_length = encoding_length
        self.max_prefix_length = max_prefix_length
        self.q_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.k_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.v_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.attention = nn.MultiheadAttention(2 * self.hidden_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(2 * self.hidden_dim * self.max_prefix_length, self.output_dim)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.ln = nn.LayerNorm(2 * self.hidden_dim * self.max_prefix_length)

    def forward(self, X):  # X:[32, 21, 100]

        q = self.q_fc(X)
        k = self.k_fc(X)
        v = self.v_fc(X)
        output, output_weights = self.attention(q, k, v)  # [32, 21, 100]
        # output = torch.sum(output, 1)  # [32, 100]
        output = self.flatten(output + X)
        out = F.relu(self.ln(output))
        # out = self.dropout(out)
        out = self.fc(out)  # [32, 3]

        return out


class ResAttSelf(nn.Module):  # 模型参数数量: 216734, 216K

    def __init__(self, encoding_length, label_num, max_prefix_length, att_channel, hidden_dim=256, epochs=100,
                 batch_size=32, num_layers=2, batch_first=True, bidirectional=True, dropout_rate=0.5,
                 save_path='net3.pth'):
        super(ResAttSelf, self).__init__()
        self.encoding_length = encoding_length
        self.label_num = label_num  # 标签的种类, 即目标活动的数量
        self.att_channel = att_channel
        self.max_prefix_length = max_prefix_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.save_path = save_path
        self.attention = SelfAttention(self.batch_size, self.max_prefix_length, self.hidden_dim, 128,
                                       encoding_length=self.encoding_length)
        self.lstm = nn.LSTM(input_size=self.att_channel * self.encoding_length, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.res_block = ResBlock(self.att_channel, 128)
        self.linear = nn.Linear(256, self.label_num)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        :param x: tensor, [batch, seq_len, input_dim]=[batch,max_prefix_length,encoding_length],[32,73,3,59]
        :return: out, tensor, [batch, seq_len]
        """
        x1 = torch.flatten(x, 2, 3)
        weight1 = next(self.parameters()).data
        hidden1 = (weight1.new(2 * self.num_layers, self.batch_size, self.hidden_dim).zero_().float().cuda(),
                   weight1.new(2 * self.num_layers, self.batch_size, self.hidden_dim).zero_().float().cuda())
        y1, _ = self.lstm(x1, hidden1)
        out1 = self.attention(y1)
        out2 = self.res_block(x.permute(0, 2, 1, 3))
        out = torch.concat([out1, out2], dim=1)
        # out = self.dropout(out)
        out = self.linear(out)
        return out


def train_model(l_rate, data_train, data_valid, encoding_length, label_num, max_prefix_length, input_channel,
                model_path):
    net = ResAttSelf(encoding_length, label_num, max_prefix_length, input_channel)
    print(f'模型参数为{sum(x.numel() for x in net.parameters())}')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=l_rate)  # last = 0.08/0.1
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    net.cuda()
    labels_train, predict_train = [], []
    early_stopping = pytorchtools.EarlyStopping(patience=10, verbose=True, path=model_path)
    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [], []
    for epoch in range(net.epochs):
        net.train()
        for inputs, labels in data_train:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = net(inputs)
            train_loss = criterion(output, labels.long())
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # 计算准确度
            predict = torch.nn.Softmax(dim=1)(output)
            predict = torch.argmax(predict, 1)
            labels_train += labels.tolist()
            predict_train += predict.tolist()

        # 测试
        net.eval()
        labels_valid, predict_valid, pred_probs = [], [], []
        with torch.no_grad():  # 表示不需要保存训练参数过程中的梯度
            for inputs_, labels_ in data_valid:
                if USE_CUDA:
                    inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                test_output = net(inputs_)
                val_loss = criterion(test_output.squeeze(), labels_.long())
                # scheduler.step(val_loss)
                valid_losses.append(val_loss.item())
                # 计算准确度
                predict_output = torch.nn.Softmax(dim=1)(test_output)
                predict_test = torch.argmax(predict_output, 1)
                labels_valid += labels_.tolist()
                predict_valid += predict_test.tolist()
                pred_probs += predict_output.tolist()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_acc = metrics.accuracy_score(labels_train, predict_train)
        valid_acc = metrics.accuracy_score(labels_valid, predict_valid)

        print("| Epoch: {:2d}/{:2d} |".format(epoch + 1, net.epochs),
              "Train Loss: {:.4f} |".format(train_loss),
              "Valid Loss: {:.4f} |".format(valid_loss),
              "Train Acc: {:.4f} |".format(train_acc),
              "Valid Acc: {:.4f} |".format(valid_acc),
              )
        # 清空loss列表
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def run_test(data_test, encoding_length, label_num, max_prefix_length, input_channel, write_add, model_path):
    net = ResAttSelf(encoding_length, label_num, max_prefix_length, input_channel)
    net.load_state_dict(torch.load(model_path))
    print(f'模型参数为{sum(x.numel() for x in net.parameters())}')
    criterion = nn.CrossEntropyLoss()
    net.cuda()
    net.eval()
    labels_test, predict_test = [], []
    test_losses = []
    predict_probs = []
    with torch.no_grad():  # 表示不需要保存训练参数过程中的梯度
        for inputs_, labels_ in data_test:
            if USE_CUDA:
                inputs_, labels_ = inputs_.cuda(), labels_.cuda()
            test_output = net(inputs_)
            val_loss = criterion(test_output.squeeze(), labels_.long())
            test_losses.append(val_loss.item())
            # 计算准确度
            predict_output = torch.nn.Softmax(dim=1)(test_output)
            predict_ = torch.argmax(predict_output, 1)

            predict_probs += predict_output.tolist()
            labels_test += labels_.tolist()
            predict_test += predict_.tolist()

    test_loss = np.average(test_losses)
    test_acc = metrics.accuracy_score(labels_test, predict_test)
    test_macro_pr = metrics.precision_score(labels_test, predict_test, average='macro', zero_division=0)
    test_macro_re = metrics.recall_score(labels_test, predict_test, average='macro', zero_division=0)
    test_macro_f = metrics.f1_score(labels_test, predict_test, average='macro', zero_division=0)
    label_probs = trans_to_prob(labels_test, label_num)
    test_micro_auc = metrics.roc_auc_score(label_probs, predict_probs, average='micro')

    with open(write_add, 'a', encoding='utf8') as f:
        f.write(
            f'{test_loss},{test_acc},{test_macro_pr},{test_macro_re},{test_macro_f},{test_micro_auc}\n')
    f.close()

    print(f'results ===>>> '
          f'test_loss: {test_loss} | test_acc: {test_acc} | test_macro_pr: {test_macro_pr} | '
          f'test_macro_re: {test_macro_re} | test_macro_f: {test_macro_f} | test_micro_auc: {test_micro_auc}\n')


if __name__ == '__main__':
    print("结果如下：")

    parameters = []
    with open('parameters.csv', 'r', encoding='utf8') as f:
        for line in csv.reader(f):
            parameters.append(line)
    f.close()
    for line in parameters:
        print(f"****************************数据集为{line[0]}****************************\n")
        print(f"****************************数据集为{line[0]}****************************\n")
        print(f"****************************数据集为{line[0]}****************************\n")

        # 参数设置
        data_address = line[0]
        case_num = int(line[1])
        attributes_ = line[2:]
        model_name = 'dual_view'

        # 其他参数计算
        label_attribute_ = line[2]
        input_channel_ = len(attributes_)  # len(attributes_)
        model_path_ = model_name + '_' + re.findall(r'.+?\.', data_address)[0][:-1] + '.pth'
        write_address = 'result_' + re.findall(r'.+?\.', data_address)[0][:-1] + '_' + model_name + '.csv'
        with open(write_address, 'a', encoding='utf8') as f:
            f.write('test_loss,test_acc,test_macro_pr,test_macro_re,test_macro_f,test_micro_auc\n')
        f.close()

        # 数据预处理
        log = CsvEventLog(data_address, attributes=attributes_, label_attribute=label_attribute_, row_num=case_num)
        encoding_length_ = log.encoding_length
        max_prefix_length_ = log.max_length - 1
        label_num_ = len(log.label_encoding)

        train_log, test_log= dataset_split(log.log, train_percent=0.8)
        # 拆分数据集（k-fold法）
        kf = KFold(n_splits=5)
        for k, (fold_train_index, valid_index) in enumerate(kf.split(train_log)):  # 第k个fold

            print()
            print(f"-------------------------第{k + 1}次实验-------------------------")
            print(f"-------------------------第{k + 1}次实验-------------------------")
            print(f"-------------------------第{k + 1}次实验-------------------------")
            print()

            fold_train_log = [train_log[i] for i in fold_train_index]
            valid_log = [train_log[i] for i in valid_index]

            x_train_set, y_train_set = log.dataset_encoding(fold_train_log)
            x_valid_set, y_valid_set = log.dataset_encoding(valid_log)
            x_test_set, y_test_set = log.dataset_encoding(test_log)

            del fold_train_log, valid_log
            gc.collect()

            # 训练
            print("-------------------------转换为tensor dataset数据-------------------------")
            train_data = TensorDataset(torch.tensor(x_train_set, dtype=torch.float),
                                       torch.tensor(y_train_set, dtype=torch.float))
            # torch.save(train_data, 'train_data.pt')
            valid_data = TensorDataset(torch.tensor(x_valid_set, dtype=torch.float),
                                       torch.tensor(y_valid_set, dtype=torch.float))
            # torch.save(valid_data, 'valid_data.pt')
            test_data = TensorDataset(torch.tensor(x_test_set, dtype=torch.float),
                                      torch.tensor(y_test_set, dtype=torch.float))
            # torch.save(test_data, 'test_data.pt')
            del x_train_set, y_train_set, x_valid_set, y_valid_set, x_test_set, y_test_set
            gc.collect()

            print("-------------------------转换为loader数据-------------------------")
            # train_data = torch.load('train_data.pt')
            train_loader = DataLoader(train_data, shuffle=True, batch_size=32, drop_last=True)
            # valid_data = torch.load('valid_data.pt')
            valid_loader = DataLoader(valid_data, shuffle=True, batch_size=32, drop_last=True)

            del train_data, valid_data
            gc.collect()

            print("-------------------------训练模型-------------------------")
            if USE_CUDA:
                print('Run on GPU.')
            else:
                print('No GPU available, run on CPU.')
            train_model(l_rate=0.0001,
                        data_train=train_loader,
                        data_valid=valid_loader,
                        encoding_length=encoding_length_,
                        max_prefix_length=max_prefix_length_,
                        label_num=label_num_,
                        input_channel=input_channel_,
                        model_path=model_path_
                        )

            print("-------------------------测试模型-------------------------")
            # x2 = torch.load('x.pt')
            test_loader = DataLoader(test_data, shuffle=True, batch_size=32, drop_last=True)
            run_test(test_loader,
                     encoding_length=encoding_length_,
                     max_prefix_length=max_prefix_length_,
                     label_num=label_num_,
                     input_channel=input_channel_,
                     write_add=write_address,
                     model_path=model_path_
                     )

