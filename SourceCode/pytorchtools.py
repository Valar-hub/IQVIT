import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class MixupDataset(Dataset):
    def __init__(self, dataset, alpha=0.2):
        self.cnt=0
        self.dataset = dataset
        self.alpha = alpha

    def __getitem__(self, index):
        self.cnt=self.cnt+1
        img1, label1 = self.dataset[index]
        r = random.randint(0, len(self.dataset) - 1)
        img2, label2 = self.dataset[r]

        lam = torch.tensor(random.betavariate(self.alpha, self.alpha))
        

        if self.cnt % 3!=0:
           lam=torch.tensor(1).float()
        img = lam * img1 + (1 - lam) * img2
        
        return img, label1,label2,lam

    def __len__(self):
        return len(self.dataset)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
 
    def __call__(self, val_loss, model):
 
        score = -val_loss
 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
 
    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth') # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl') # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss



def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    total_params = total_params*4/1024/1024
    print('{:.2f} MB total parameters.'.format(total_params))
    
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

