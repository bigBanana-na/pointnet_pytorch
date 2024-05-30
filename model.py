from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 3D空间变换网络 (STN3d)
# 该网络的主要功能是对输入的点云进行3x3仿射变换，以便对输入数据进行标准化处理，增强模型的鲁棒性。
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        # 定义三个一维卷积层，输入分别是3维点，输出为64、128和1024维特征
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 定义三个全连接层，将1024维特征最终映射到9维，用于生成3x3的变换矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        # 定义批标准化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # 卷积 -> 批标准化 -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 最大池化操作，获取全局特征
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # 全连接层 -> 批标准化 -> ReLU
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 加入单位矩阵，以保持输入点云的原始变换信息
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# k维特征空间变换网络 (STNkd)
# 该网络的主要功能是对输入特征进行k x k的仿射变换，通常用于特征空间的标准化。
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        # 定义三个一维卷积层，输入为k维特征，输出分别为64、128和1024维特征
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 定义三个全连接层，将1024维特征最终映射到k*k维，用于生成k x k的变换矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        # 定义批标准化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        # 卷积 -> 批标准化 -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 最大池化操作，获取全局特征
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # 全连接层 -> 批标准化 -> ReLU
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 加入单位矩阵，以保持输入特征的原始变换信息
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

# PointNet特征提取网络
# 该网络用于从点云中提取全局和局部特征。
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        # 使用STN3d进行输入点云的空间变换
        self.stn = STN3d()
        # 定义三个一维卷积层，输入为3维点，输出分别为64、128和1024维特征
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 定义批标准化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        # 获取空间变换矩阵
        trans = self.stn(x)
        x = x.transpose(2, 1)
        # 对输入点云应用空间变换
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            # 对特征应用变换
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # 最大池化操作，获取全局特征
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# PointNet分类网络
# 该网络用于对点云进行分类任务，通过全连接层将特征映射到类别空间。
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        # 使用PointNetfeat进行特征提取
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        # 定义三个全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        # 定义批标准化层
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 获取特征和变换矩阵
        x, trans, trans_feat = self.feat(x)
        # 全连接层 -> 批标准化 -> ReLU -> Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

# PointNet密集分类网络（用于分割任务）
# 该网络用于对点云进行逐点分类任务，通过卷积层对每个点进行分类。
class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        # 使用PointNetfeat进行特征提取
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        # 定义四个一维卷积层，最后一层将特征映射到类别空间
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        # 定义批标准化层
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # 获取特征和变换矩阵
        x, trans, trans_feat = self.feat(x)
        # 卷积 -> 批标准化 -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

# 特征变换正则化器
# 用于正则化特征变换矩阵，确保变换矩阵接近正交，以减少特征变换中的畸变。
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    # 计算变换矩阵与单位矩阵之间的差异，并求其Frobenius范数作为正则化损失
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

# 模块测试
if __name__ == '__main__':
    # 测试STN3d模块
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    # 测试STNkd模块
    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    # 测试PointNetfeat模块（提取全局特征）
    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    # 测试PointNetfeat模块（提取局部特征）
    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    # 测试PointNetCls模块（分类任务）
    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    # 测试PointNetDenseCls模块（分割任务）
    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
