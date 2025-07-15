# -*- coding: utf-8 -*- 
# @Time : 2024/9/27 15:53 
# @Author : binb_chen@163.com
# @File : dual_view_Receipt_XAI.py
# encoding=utf8

import math
import os
import random
import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from gensim.models import Word2Vec

DATASET = 'BPIC12_W'
ATTRIBUTES = ("concept:name", "org:resource", "lifecycle:transition", "time_from_start_category",
              "time_from_last_category")  # 第一个位置放活动属性名
ENCODING_MODE = 'embedding'  # choose 'embedding', 'one_hot', or 'w2v'
GPU_ID = 1
RANDOM_SEED = 1
FOLD = 5

EPOCH = 200
ENCODING_LENGTH = 32
# batch_size: 64
# learning_rate: 4.8041526483943725e-05
# drop_rate: 0.2557492714957794
# hidden_dim: 384
BATCH_SIZE = 64
LEARNING_RATE = 4.8041526483943725e-05
DROP_RATE = 0.2557492714957794
HIDDEN_DIM = 384

DATA_ADD = "dataset/" + DATASET + ".csv"
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.set_device(GPU_ID)  # 指定GPU


def trans_to_str(val):
    if isinstance(val, float):
        new_val = str(int(val))
    elif isinstance(val, int):
        new_val = str(val)
    else:
        new_val = val
    return new_val


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def read_log(log_add, case_num_):
    log = [[] for i in range(case_num_)]
    with open(log_add, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            log[int(row['case']) - 1].append(row)
    f.close()
    return log


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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='resAttention.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CsvEventLog:
    """
    用来处理xes文件事件日志
    """

    def __init__(self, log_add=DATA_ADD, attributes=ATTRIBUTES, encoding_length=ENCODING_LENGTH,
                 encoding_mode=ENCODING_MODE, min_length=3):
        """

        :param log_add: str, log address
        :param attributes: list, chose attributes
        :param encoding_length: int, encoding length
        :param encoding_mode: str, choose 'embedding', 'one_hot', or 'w2v'
        :param min_length: int, 最短的trace长度, 包括下一个事件, 低于self.min_length的都会被忽略
        """

        self.log_df = pd.read_csv(log_add)
        self.case_num = self.log_df.iloc[-1].iloc[0]
        self.log_dict = read_log(log_add, self.case_num)  # 事件日志, [[dict,dict,...],[],...,[]]
        self.min_length = min_length  # 最短的trace长度, 包括下一个事件, 低于self.min_length的都会被忽略
        self.max_length = self.log_df['case'].value_counts().max()  # 最长的trace长度, 用来确定事件级LSTM的输入维度
        self.max_prefix_length = self.max_length - 1
        self.attributes = attributes  # list, 所选属性
        self.label_attribute = attributes[0]
        self.attributes_values, self.attributes_values_nums = dict(), dict()
        for attribute in self.attributes:
            self.attributes_values[attribute] = list(set(self.log_df[attribute].values.tolist()))
            self.attributes_values_nums[attribute] = len(self.attributes_values[attribute])

        self.encoding_length = encoding_length
        self.encoding_mode = encoding_mode

        # label编码
        self.label_encoding = dict()
        for i in range(len(self.attributes_values[self.label_attribute])):
            self.label_encoding[self.attributes_values[self.label_attribute][i]] = i  # 事件(concept:name)标记结果

        # prefix编码
        self.attribute_encodings = dict()

        if self.encoding_mode == 'embedding':
            for attribute in self.attributes:
                self.attribute_encodings[attribute] = self.embedding(attribute)  # 时间属性编码结果

        elif self.encoding_mode == 'w2v':
            self.trace_lists = self.corpus_build()  # 语料库构建
            self.w2v_model_build()  # 训练word2vec模型
            for attribute in self.attributes:
                self.attribute_encodings[attribute] = self.w2v_encoding(attribute)  # 时间属性编码结果

        else:  # one_hot
            self.max_atts = max([len(att_list) for att_list in self.attributes_values.values()])
            self.encoding_length = self.max_atts if self.max_atts > 10 else 10  # one-hot模式下属性编码的长度
            for attribute in self.attributes:
                self.attribute_encodings[attribute] = self.one_hot_encoding(attribute)

    def corpus_build(self):

        trace_lists = {att: [] for att in self.attributes}  # 按属性分开
        log_df = self.log_df.astype(str)
        for case_number, group in log_df.groupby('case'):
            for attribute in self.attributes:
                trace_lists[attribute].append(group[attribute].tolist())

        return trace_lists

    def w2v_model_build(self, train_epoch=10):
        w2v_add = 'w2v_model/'
        mkdir(w2v_add)
        for attribute in self.attributes:
            w2v_model = Word2Vec(vector_size=self.encoding_length, seed=RANDOM_SEED, sg=0, min_count=1, workers=1)
            w2v_model.build_vocab(self.trace_lists[attribute], min_count=1)
            total_examples = w2v_model.corpus_count
            w2v_model.train(self.trace_lists[attribute], total_examples=total_examples, epochs=train_epoch)
            model_save_path = f'{w2v_add}{DATASET}_{attribute}_w2v_model.h5'
            w2v_model.save(model_save_path)

    def w2v_encoding(self, attribute_name):

        model_save_path = f'w2v_model/{DATASET}_{attribute_name}_w2v_model.h5'
        vec_model = Word2Vec.load(model_save_path)
        encoding_result = dict()
        for val in self.attributes_values[attribute_name]:
            new_val = trans_to_str(val)
            encoding_result[new_val] = vec_model.wv[new_val]

        return encoding_result

    def one_hot_encoding(self, attribute_name):
        """
        对目标属性进行one-hot编码，并以字典形式返回编码结果，时间戳除外
        :param attribute_name: str, 目标属性名称
        :return:encoding_result, dict, 对应属性的每一个值的编码结果，以字典形式输出，例如：{'org:resource': 00010, ...}
        """
        encoding_result = dict()

        for i in range(len(self.attributes_values[attribute_name])):
            init_encoding = [0 for i in range(self.encoding_length)]  # 初始化编码
            init_encoding[- (i + 1)] = 1
            encoding_result[self.attributes_values[attribute_name][i]] = init_encoding

        return encoding_result

    def embedding(self, attribute_name):
        attribute_num = self.attributes_values_nums[attribute_name]
        num_list = torch.IntTensor([i for i in range(attribute_num)])
        embedding = nn.Embedding(attribute_num, self.encoding_length)
        results = embedding(num_list)
        embedding_result = dict()
        for j in range(attribute_num):
            embedding_result[self.attributes_values[attribute_name][j]] = results[j].tolist()
        return embedding_result

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
                if a_trace[i][attribute] in self.attributes_values[attribute]:
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
        print("-----------log编码-----------")
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
        # self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x):
        y1 = self.res_net1(x)  # 32,64,7,11
        y2 = self.res_net2(y1)  # 32, 128, 4, 6
        y3 = self.res_net3(y2)  # 32, 256, 2, 3
        y4 = self.res_net4(y3)  # 32, 512, 1, 2
        out = torch.squeeze(self.pooling(y4), (2, 3))  # 32, 512
        out = F.relu(self.linear(out))

        return out


class SelfAttention(nn.Module):
    def __init__(self, max_prefix_length, hidden_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_prefix_length = max_prefix_length

        self.q_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.k_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.v_fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.attention = nn.MultiheadAttention(2 * self.hidden_dim, num_heads=2, batch_first=True)
        self.linear = nn.Linear(2 * self.hidden_dim * self.max_prefix_length, self.output_dim)
        self.flatten = nn.Flatten()
        self.ln = nn.LayerNorm(2 * self.hidden_dim * self.max_prefix_length)

    def forward(self, X):  # X:[32, 21, 100]

        q = self.q_fc(X)
        k = self.k_fc(X)
        v = self.v_fc(X)
        output, output_weights = self.attention(q, k, v)  # [32, 21, 100]
        output = self.flatten(output + X)
        out = self.ln(output)
        out = F.relu(self.linear(out))  # [32, 3]

        return out


class ResAttSelf(nn.Module):
    def __init__(self, encoding_length, label_num, max_prefix_length, batch_size, attribute_num=len(ATTRIBUTES),
                 hidden_dim=HIDDEN_DIM, dropout_rate=DROP_RATE):
        super(ResAttSelf, self).__init__()
        self.encoding_length = encoding_length
        self.label_num = label_num  # 标签的种类, 即目标活动的数量
        self.attribute_channel = attribute_num
        self.max_prefix_length = max_prefix_length
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.attention = SelfAttention(self.max_prefix_length, self.hidden_dim, self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.attribute_channel * self.encoding_length, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.res_block = ResBlock(self.attribute_channel, self.hidden_dim)
        self.linear = nn.Linear(2 * self.hidden_dim, self.label_num)
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
        out_lstm, _ = self.lstm(x1, hidden1)
        out_sfem = self.attention(out_lstm)
        out_ffem = self.res_block(x.permute(0, 2, 1, 3))
        out = torch.concat([out_sfem, out_ffem], dim=1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def run_test(encoding_length, label_num, max_prefix_length, batch_size, data_test, model_path, write_add):
    net = ResAttSelf(encoding_length, label_num, max_prefix_length, batch_size)
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
            val_loss = criterion(test_output, labels_.long())
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
    test_macro_g = geometric_mean_score(labels_test, predict_test, average="macro")

    with open(write_add, 'a', encoding='utf8') as f:
        f.write(
            f'{test_loss},{test_acc},{test_macro_pr},{test_macro_re},{test_macro_f},{test_macro_g}\n')
    f.close()

    print(f'results ===>>> '
          f'test_loss: {test_loss} | test_acc: {test_acc} | test_macro_pr: {test_macro_pr} | '
          f'test_macro_re: {test_macro_re} | test_macro_f: {test_macro_f} | test_macro_g: {test_macro_g}\n')


def data_pro():
    log = CsvEventLog()
    label_num = len(log.label_encoding)
    max_prefix_length = log.max_prefix_length
    encoding_length = log.encoding_length

    # 划分训练集和测试集
    train_log, test_log = dataset_split(log.log_dict, train_percent=0.8)

    x_train_set, y_train_set = log.dataset_encoding(train_log)
    x_test_set, y_test_set = log.dataset_encoding(test_log)

    data_dir = 'pro_data/'
    mkdir(data_dir)
    x_train_data_add = f'{data_dir}{DATASET}_{ENCODING_MODE}_train_data_x.npy'
    x_test_data_add = f'{data_dir}{DATASET}_{ENCODING_MODE}_test_data_x.npy'
    y_train_data_add = f'{data_dir}{DATASET}_{ENCODING_MODE}_train_data_y.npy'
    y_test_data_add = f'{data_dir}{DATASET}_{ENCODING_MODE}_test_data_y.npy'

    np.save(x_train_data_add, np.array(x_train_set))
    np.save(x_test_data_add, np.array(x_test_set))
    np.save(y_train_data_add, np.array(y_train_set))
    np.save(y_test_data_add, np.array(y_test_set))

    parameter_add = f'{data_dir}{DATASET}_{ENCODING_MODE}_parameters.csv'
    parameters = {
        'encoding_length': encoding_length,
        'label_num': label_num,
        'max_prefix_length': max_prefix_length,
    }
    parameters_df = pd.DataFrame(parameters, index=[0])  # 添加index以确保每一行都对应
    parameters_df.to_csv(parameter_add, index=False)


def get_parameters():
    parameters_df = pd.read_csv(f'pro_data/{DATASET}_{ENCODING_MODE}_parameters.csv')
    parameters = parameters_df.to_dict(orient='records')[0]

    return parameters


def main():
    # 下载关键参数
    parameters = get_parameters()
    # 设置测试集输出地址
    test_write_add = f"{os.path.basename(__file__)[:-3]}_{ENCODING_MODE}_result.csv"
    with open(test_write_add, 'a', encoding='utf-8') as f:
        f.write(f'----------{RANDOM_SEED}----------\n')
        f.write(
            f'batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, dropout_rate={DROP_RATE}, hidden_dim={HIDDEN_DIM}\n')
        f.write('test_loss,test_acc,test_macro_pr,test_macro_re,test_macro_f,test_macro_g\n')
    f.close()

    net = ResAttSelf(parameters['encoding_length'], parameters['label_num'], parameters['max_prefix_length'],
                     BATCH_SIZE)
    print(f'模型参数为{sum(x.numel() for x in net.parameters())}')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)  # last = 0.08/0.1

    # 划分训练集和测试集
    train_data_add_x = f'pro_data/{DATASET}_{ENCODING_MODE}_train_data_x.npy'
    train_data_add_y = f'pro_data/{DATASET}_{ENCODING_MODE}_train_data_y.npy'
    x_train_data = torch.tensor(np.load(train_data_add_x), dtype=torch.float)
    y_train_data = torch.tensor(np.load(train_data_add_y), dtype=torch.float)

    avg_valid_losses = []
    # 拆分数据集（k-fold法）
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=RANDOM_SEED)
    for k, (train_index, valid_index) in enumerate(kf.split(np.load(train_data_add_y))):  # 第k个fold

        print()
        print(f"-------------------------第{k + 1}次实验-------------------------")
        print(f"-------------------------第{k + 1}次实验-------------------------")
        print(f"-------------------------第{k + 1}次实验-------------------------")
        print()

        x_train_data_fold = x_train_data[torch.tensor(train_index).int()]
        x_valid_data_fold = x_train_data[torch.tensor(valid_index).int()]
        y_train_data_fold = y_train_data[torch.tensor(train_index).int()]
        y_valid_data_fold = y_train_data[torch.tensor(valid_index).int()]
        train_tensor_dataset = TensorDataset(x_train_data_fold, y_train_data_fold)
        valid_tensor_dataset = TensorDataset(x_valid_data_fold, y_valid_data_fold)
        train_loader = DataLoader(train_tensor_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
        valid_loader = DataLoader(valid_tensor_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

        print("-------------------------训练模型-------------------------")
        if USE_CUDA:
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        fold_model_dir = 'fold_model/'
        mkdir(fold_model_dir)
        model_path_fold = f'{fold_model_dir}{DATASET}_{ENCODING_MODE}_{os.path.basename(__file__)[:-3]}_{k + 1}.pth'

        net.cuda()
        labels_train, predict_train = [], []
        early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path_fold)
        train_losses, valid_losses = [], []
        for epoch in range(EPOCH):
            net.train()
            for inputs, labels in train_loader:
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
                for inputs_, labels_ in valid_loader:
                    if USE_CUDA:
                        inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                    valid_output = net(inputs_)
                    val_loss = criterion(valid_output, labels_.long())
                    valid_losses.append(val_loss.item())
                    # 计算准确度
                    predict_output = torch.nn.Softmax(dim=1)(valid_output)
                    predict_test = torch.argmax(predict_output, 1)
                    labels_valid += labels_.tolist()
                    predict_valid += predict_test.tolist()
                    pred_probs += predict_output.tolist()

            avg_train_loss = np.average(train_losses)
            avg_valid_loss = np.average(valid_losses)
            avg_valid_losses.append(avg_valid_loss)
            train_acc = metrics.accuracy_score(labels_train, predict_train)
            valid_acc = metrics.accuracy_score(labels_valid, predict_valid)
            print("| Epoch: {:2d}/{:2d} |".format(epoch + 1, EPOCH),
                  "Train Loss: {:.4f} |".format(avg_train_loss),
                  "Valid Loss: {:.4f} |".format(avg_valid_loss),
                  "Train Acc: {:.4f} |".format(train_acc),
                  "Valid Acc: {:.4f} |".format(valid_acc),
                  )
            # 清空loss列表
            train_losses = []
            valid_losses = []

            early_stopping(avg_valid_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("-------------------------测试模型-------------------------")
        # 下载测试数据
        test_data_add_x = f'pro_data/{DATASET}_{ENCODING_MODE}_test_data_x.npy'
        test_data_add_y = f'pro_data/{DATASET}_{ENCODING_MODE}_test_data_y.npy'
        x_test_data = np.load(test_data_add_x)
        y_test_data = np.load(test_data_add_y)
        test_tensor_dataset = TensorDataset(torch.tensor(x_test_data, dtype=torch.float),
                                            torch.tensor(y_test_data, dtype=torch.float))
        test_loader = DataLoader(test_tensor_dataset)

        run_test(parameters['encoding_length'],
                 parameters['label_num'],
                 parameters['max_prefix_length'],
                 batch_size=1,
                 data_test=test_loader,
                 model_path=model_path_fold,
                 write_add=test_write_add)


if __name__ == '__main__':

    data_pro()
    main()
