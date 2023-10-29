import sys
from torch import nn
from tqdm import tqdm
import torch
from FL import focal_loss as loss_function
import torch.nn.functional as F
from multi_train_utils.distributed_utils import reduce_value, is_main_process

class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()

    def forward(self, y, y_true):
        epsilon = 1e-7  # 避免分母为零的情况
        _, y_pred = torch.max(y, dim=1)
        tp = torch.sum(torch.round(y_pred * y_true))  # 计算真阳性
        fp = torch.sum(torch.round(y_pred) - y_true == 1)  # 计算假阳性
        fn = torch.sum(y_pred - torch.round(y_true) == 1)  # 计算假阴性
        precision = tp / (tp + fp + epsilon)  # 计算准确率
        recall = tp / (tp + fn + epsilon)  # 计算召回率
        f1_score = 2 * precision * recall / (precision + recall + epsilon)  # 计算f1分数
        loss = 1 - f1_score  # 定义f1分数损失函数
        return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    #loss_function = focal_loss()#torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    #loss_function = nn.CrossEntropyLoss()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        #images, labels1,labels2,lam= data
        images, labels =data
        images=images.permute(0,2,1,3).contiguous()
        pred = model(images.to(device))
        loss=nn.CrossEntropyLoss()
        #loss=F1Loss()
        loss=loss(pred,labels.to(device))
        loss.requires_grad_(True)
        #loss = loss_function(pred, labels1.to(device),labels2.to(device),lam.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    mean_loss = torch.zeros(1).to(device)  
    #loss_function = focal_loss()#torch.nn.CrossEntropyLoss()
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        images=images.permute(0,2,1,3).contiguous()
        pred = model(images.to(device))
        
        loss=nn.CrossEntropyLoss()
        #val_loss
        loss = loss(pred,labels.to(device))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(),mean_loss.item()






