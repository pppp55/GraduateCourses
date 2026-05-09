import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'      # 验证集
        self.test_path = dataset + '/data/test.txt'    # 测试集
        # 类别名单
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt',
            encoding='utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl' # 词表
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + \
                         self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')   # 设备
        self.dropout = 0.5           # 随机失活
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)       # 类别数
        self.n_vocab = 0                          # 词表大小，在运行时赋值
        self.num_epochs = 6                  # epoch数
        self.batch_size = 128                # mini-batch大小
        self.pad_size = 32                   # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4            # 学习率
        self.embed = 300           # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5    # 多头的数量
        self.num_encoder = 2 # encoder的层数


# Transformer网络结构
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(
            config.n_vocab, config.embed,
            padding_idx=config.n_vocab - 1
        )
        # 位置编码
        self.postion_embedding = Positional_Encoding(
            config.embed, config.pad_size,
            config.dropout, config.device
        )
        # encoder结构
        self.encoder = Encoder(config.dim_model, config.num_head,
                               config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])
        # 全连接
        self.fc1 = nn.Linear(
            config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        # 输入数据送入到词嵌入层
        out = self.embedding(x[0])
        # 位置编码
        out = self.postion_embedding(out)
        # encoder层
        for encoder in self.encoders:
            out = encoder(out)
        # encoder的输出
        out = out.view(out.size(0), -1)
        # 全连接
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        # 多头注意力机制
        self.attention = Multi_Head_Attention(dim_model,
                                              num_head, dropout)
        #  前馈网络
        self.feed_forward = Position_wise_Feed_Forward(dim_model,
                                                       hidden, dropout)

    def forward(self, x):
        # 将数据绑定到多头注意力中
        out = self.attention(x)
        # 前向计算过程
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos /
                                 (10000.0 **
                                  (i // 2 * 2.0 / embed))
                                 for i in range(embed)]
                                for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 构建残差网络
        out = x + nn.Parameter(
            self.pe, requires_grad=False).to(self.device)
        # 放弃神经元，防止过拟合
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        # 计算attention中Q和K的乘积
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        # 计算attention得分
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        # Q、K、V矩阵的计算
        self.fc_Q = nn.Linear(dim_model,
                              num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model,
                              num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model,
                              num_head * self.dim_head)
        # 缩放点积的注意力
        self.attention = Scaled_Dot_Product_Attention()
        # 全连接
        self.fc = nn.Linear(num_head * self.dim_head,
                            dim_model)
        self.dropout = nn.Dropout(dropout)
        # layer normalization
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)
        K = K.view(batch_size * self.num_head, -1,
                   self.dim_head)
        V = V.view(batch_size * self.num_head, -1,
                   self.dim_head)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1,
                               self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)  # 全连接层1
        self.fc2 = nn.Linear(hidden, dim_model)  # 全连接层2
        self.dropout = nn.Dropout(dropout)   # dropout层
        self.layer_norm = nn.LayerNorm(dim_model) # layer Normalization

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out) # layer Normalization
        return out
