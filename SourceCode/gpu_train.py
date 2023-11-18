import os
import math
import tempfile
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PVQNet import vit_demo
from VGG import VGG_diy
from deit import deit_tiny_patch16_LS
from fitsFolderV1 import get_purify_fits_folder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import read_split_data, plot_data_loader_image
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

from pytorchtools import EarlyStopping,MixupDataset,cnn_paras_count

def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    images_path=args.data_path

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}

    # 实例化训练数据集
    if args.use_mixup==True:
        train_data_set = get_purify_fits_folder(images_path+'/train/')

    else:
        train_data_set = get_purify_fits_folder(images_path+'/train/')
     
    # 实例化验证数据集
    val_data_set = get_purify_fits_folder(images_path+'/val/')

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)
    
    # 转为DDP模型
    #model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])#,find_unused_parameters=True)#,find_unused_parameters=True,device_ids=[args.gpu])
    # optimizer
    # 8e-1,5e-1,
    #lrs=[1e-1,9e-2,4e-2,1e-2,8e-3,4e-3,5*1e-4,1e-4,5*1e-5,1e-5]
    #lrs=[8e-2,2e-2,8e-3,2e-3,8e-4]
    lrs=[1e-4,5*1e-5,2*1e-5,8*1e-6,5*1e-6,8*1e-7]
    for lr in lrs:
        
         # 实例化模型
        if args.model=='VIT':
            model = vit_demo().to(device)
        elif args.model=='VGG':
            model = VGG_diy().to(device)
        elif args.model=='Deit':
            model =deit_tiny_patch16_LS().to(device)
        print('model size is {}'.format(cnn_paras_count(model)))
        #model = nn.parallel.DistributedDataParallel(model)
        #model = model.to(device)

        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])
        #model = model.double()

        # 如果存在预训练权重则载入
        if os.path.exists(weights_path):
            weights_dict = torch.load(weights_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
            print(load_weights_dict)
            model.load_state_dict(load_weights_dict, strict=True)
        else:
            checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
            # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
            if rank == 0:
                torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        #model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # 是否冻结权重
        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除最后的全连接层外，其他权重全部冻结
                if "fc" not in name:
                    para.requires_grad_(False)
        else:
            # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
            if args.syncBN:
                # 使用SyncBatchNorm后训练会更耗时
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

        if rank == 0:
            print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/'.format(("./log/{}".format(str(lr)))))
            tb_writer = SummaryWriter(log_dir=("./log/{}".format(str(lr))))
        pg = [p for p in model.parameters() if p.requires_grad]
        #optimizer = optim.SGD(pg, lr=lr,momentum=0.7,weight_decay=0.01) #0.05
        optimizer = optim.Adam(pg, lr=lr,weight_decay=0.05)
    
        #学习率预热阶段lr起始值
        #scheduler = CosineLRScheduler(optimizer=optimizer,t_initial=100,lr_min =0.0000003,warmup_t=13,warmup_lr_init=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.9, patience=8, verbose=True)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / args.epochs) * (1 - args.lrf) + args.lrf  # cosine
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    


        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        for epoch in range(args.epochs):
            
            #scheduler.step(epoch)
            train_sampler.set_epoch(epoch)

            mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

            sum_num,test_loss = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
            acc = sum_num / val_sampler.total_size
            
            scheduler.step(test_loss)

            if rank == 0:
                print("[epoch {}] accuracy: {} val_loss: {} lr_scheduler: {}".format(epoch, round(acc, 3),round(test_loss, 3),round(optimizer.param_groups[0]['lr'],6)))
                tags = ["loss", "accuracy", "learning_rate","val_loss"]
                tb_writer.add_scalar(tags[0], mean_loss, epoch)
                tb_writer.add_scalar(tags[1], acc, epoch)
                tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
                tb_writer.add_scalar(tags[3], test_loss, epoch)

                early_stopping(test_loss,model)

            if early_stopping.early_stop:
                print("Early Stopping!")
                break  
            torch.save(model.state_dict(), "./weights/model-{}-{}.pth".format(epoch,str(lr)))
         
        # 删除临时缓存文件
        if rank == 0:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)  #64
    parser.add_argument('--lr', type=float, default=0.0003) #0.0001
    parser.add_argument('--lrf', type=float, default=0.1) #0.1
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--model', type=str,default='VIT',help='choose the modelType eg.VGG PVQNet')

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default='./new_res2/')

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='./weights/save.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--mode', default='train', help='whether to train or evaluate the model')
    parser.add_argument('--log', default='./log/', help='save tensorboard logs')
    parser.add_argument('--patience', default='45',type=int, help='earlystoping patience')
    parser.add_argument('--use-mixup', default='True',type=bool, help='whether to use mixup to train the model')
    opt = parser.parse_args()


    main(opt)

