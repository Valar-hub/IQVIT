import warnings
import torch
import os
import numpy as np
from torch import nn
from fitsFolderV1 import get_path_fits_folder
from tqdm import tqdm
from VGG import VGG_diy 
from PVQNet import vit_demo
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from astropy.visualization import make_lupton_rgb
import sys
import shap
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from PIL import Image
def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_bytes = num_params * 4  # 一个 32 位浮点数占用 4 个字节
    num_bits = num_params * 32  # 一个 32 位浮点数占用 32 个比特
    return num_bytes/(1024**2)

def get_confusion_matrix(pred,label):
    confusion=torch.zeros((2,2))
    #pred = torch.max(pred, dim=1)[1]
    for x,y in zip(label,pred):
        confusion[x][y]=confusion[x][y]+1
    
    TP=confusion[0][0]
    FP=confusion[0][1]
    TN=confusion[1][1]
    FN=confusion[1][0]
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    a=torch.tensor([[TP,FP],
        [FN,TN]])
    return [(TP+TN)/(TP+FP+TN+FN).numpy(),TP/(TP+FP),TP/(TP+FN).numpy(),2*(Precision*Recall)/(Precision+Recall).numpy(),TP/(TP+TN),FP/(FP+TN).numpy()],a

def weights_change(weights):
    weights_dict = {}
    for k, v in weights.items():
      new_k = k.replace('module.', '') if 'module' in k else k
      weights_dict[new_k] = v
    return weights_dict

def filter(input_tensor,threshold):
    output_tensor = torch.zeros_like(input_tensor,device="cpu")
    output_tensor[input_tensor >= threshold] = 1
    output_tensor[input_tensor < threshold] = 0
    return output_tensor.long()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model=='VIT':
        net=vit_demo().eval()
    elif args.model=='VGG':
        net=VGG_diy().eval()
    #net = nn.DataParallel(net)
    print('model size is {:.2f}MB'.format(count_parameters(net)))
    loss_function = torch.nn.CrossEntropyLoss()
    
    image_path = args.data_path
    #reall = pd.read_csv('SDSS_res.csv')

    weights_dict=weights_change(torch.load(args.weights))
    net.load_state_dict(weights_dict,strict=True) #加载模型
    net = nn.DataParallel(net)
    net.to(device)

    test_data_set = get_path_fits_folder(image_path)#+'test')#'/final1/')
    test_iter = DataLoader(dataset=test_data_set, batch_size=1000,shuffle=False)#, collate_fn=myfunction)

    #cnt=0    
    #zs=pd.DataFrame(columns=['acc','precision','recall','f1-score','TPR','FPR'])

    tensor = torch.tensor([[0, 0],
                       [0, 0]])
    for thresh in np.arange(0.5,0.65,0.5):
        result=pd.DataFrame()
        print(thresh)
        b=[]
        c=[]
        RES=[]
        for step,(data,label) in tqdm(enumerate(test_iter)):
            #print(len(data))
            pathes=data[1]
            data=data[0]
            data=data.permute(0,3,1,2).contiguous().to(device).float()
            #target_layer=net.module.conv2
            #print(net.module)
            
            #targets=[ClassifierOutputTarget(1)]
            #cam = GradCAMPlusPlus(model=net, target_layers=target_layer, use_cuda=True)
            #grayscale_cam = cam(input_tensor=data,eigen_smooth=True,aug_smooth=True)#,eigen_smooth=True)#, targets=targets,aug_smooth=True,eigen_smooth=True)
            
            #shap_values = np.sort(shap_values, axis=(-1))[:,:, :,:]
            
            #print(type(shap_values))
            #shap_values=torch.tensor(shap_values)
            
            #print(shap_values.shape)
            #print(shap_values[0][:][:][:][:].mean(axis=(-1)))
            #print('#################################')
            #print(shap_values[1][:][:][:][:].mean(axis=(-1)))
            
            #return 
            #print(shap_values.mean(dim=(0,2,3)))
            #print(shap_values)
            pred = net(data)
            res=torch.nn.Softmax(dim=1)(pred)
            res1=res
            res1=res1.cpu().detach()
            #print(res.cpu().detach().numpy())
            res=res.cpu()
            RES.extend(res[:,1].cpu().detach().numpy())
            res=filter(res[:,1],thresh)
            #print(res.cpu().detach().numpy())
            label = torch.ones_like(res).cuda()
            _,mat=get_confusion_matrix(res,label)
            res=res.cpu()
            label=label.cpu()
            #index = np.where(res == label)[0]
            tensor = tensor + mat
            
            #background = torch.zeros(1, 5, 64, 64)
            #e = shap.DeepExplainer(net,background.cuda())
            #shap_values = np.array(e.shap_values(data))
            #shap_values = shap_values.reshape(shap_values.shape[0],shap_values.shape[1],shap_values.shape[2],64*64)
            #data1 = np.array(data.cpu())
            #np.savetxt('5_6'+'low.csv',shap_values[0][index][:][:][:].mean(axis=(-1)), delimiter=',')
            #np.savetxt('5_6'+'high.csv',shap_values[1][index][:][:][:].mean(axis=(-1)), delimiter=',')
            #np.savetxt('5_6'+'feature.csv',data1.reshape(data1.shape[0],data1.shape[1],64*64)[index].mean(axis=(-1)),delimiter=',')
           
            #new_score=pd.Series([mat[0].numpy(),mat[1].numpy(),mat[2].numpy(),mat[3].numpy(),mat[4].numpy(),mat[5].numpy()], index=zs.columns)
            #zs=zs.append(new_score,ignore_index=True)
            
            #res=filter(res[:,1],thresh)

            #data=data.cpu().numpy().reshape(data.shape[0],-1)
            #tsne = TSNE(n_components=2, random_state=0,perplexity=2,early_exaggeration=400,learning_rate=13,n_iter=10000,n_iter_without_progress=100)
            #data_tsne = tsne.fit_transform(data)
            #tsn['dim1']=data_tsne[:,0]
            #tsn['dim2']=data_tsne[:,1] 
            #print(label.detach())
            mask = torch.eq(res.to(device),label.to(device))
            indices = torch.nonzero(mask,as_tuple=False)[:,0]

            
            for i in range(len(label)):
               if i in indices:
                  b.append(pathes[i].split('/')[-1].split('_')[0])
                  new_row = pd.DataFrame([['b'+pathes[i].split('/')[-1].split('_')[0], float(res1[i,1])]], columns=['SPECOBJID', 'CONFIDENCE'])
                  result=pd.concat([result,new_row])
                  #print(pathes[i].split('/')[-1].split('_')[0])
                  #print(res1[i,1])
                    
                  
               else:
                  c.append(test_data_set.imgs[i][0].split('/')[-1].split('_')[0])
        #print(len(b))
        with open('spec.txt', "w") as file:
            for item in b:
                file.write("%s\n" % item)

        np.savetxt('RES.txt',RES)
        #print(len(c))
        #print(tensor)
            #print(get_confusion_matrix(res,label)[0])
            #print(len(c))
        #print(tensor)
            #pred=torch.max(pred, dim=1)[1]
            #print(b)        
            #print('##################################')
            #print(c)
            #for i in range(len(data)):FP /（FP + TN
            #     if pred[i] != label[i]:
            #         cnt=cnt+1
            #         z=float(test_data_set.imgs[i][0].split("_")[1])
            #         if z>4.9 and z<=5:
            #           zs.append(z)
                 #print(test_data_set.imgs[i][0])
            #print(cnt)
            #print(zs)
            #print(len(zs))
        result.to_csv('END.csv',index=False)


if __name__=='__main__':
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    #./SDSS_image/
    parser.add_argument('--data-path', type=str, default='./ZWARN4_image/')
    parser.add_argument('--model', type=str,default='FNet',help='choose the modelType eg.FNet LightVIT')
    parser.add_argument('--weights', type=str, default='./checkpoint1.pth',
                        help='initial weights path')
    parser.add_argument('--interval', type=str, default='0.1',
                        help='interval')
    #parser.add_argument('--data-path', type=str, default='./image2023V1/')
    opt = parser.parse_args()
    main(opt)
