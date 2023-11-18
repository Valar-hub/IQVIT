import torch
from torch import nn
class MixedInceptionModuleBlock(nn.Module):
    def __init__(self,inp,oup,use_5x5_conv=False):
        super().__init__()
        self.use_5x5_conv=use_5x5_conv
        self.inception1=nn.Sequential(nn.Conv2d(inp,oup,kernel_size=1,padding=0,stride=1),
                                      nn.Conv2d(oup,oup,kernel_size=3,padding=1,stride=1))
        if use_5x5_conv==True:
            self.inception2=nn.Sequential(nn.Conv2d(inp,oup,kernel_size=1,padding=0,stride=1),
                                      nn.Conv2d(oup,oup,kernel_size=5,padding=2,stride=1))
            self.bn=nn.BatchNorm2d(4*oup)
        else:
            self.bn=nn.BatchNorm2d(3*oup)

        self.inception3=nn.Sequential(nn.AvgPool2d(kernel_size=3, padding=1,stride=1),
                                     nn.Conv2d(inp,oup,kernel_size=1,padding=0,stride=1))
        self.inception4=nn.Conv2d(inp,oup,kernel_size=1,padding=0,stride=1)
        

    def forward(self,x):
        x1=self.inception1(x)
        
        #5x5 kernel
        x3=self.inception3(x) 
        x4=self.inception4(x)
        
        if self.use_5x5_conv==True:
           x2=self.inception2(x)
           x=torch.cat((x1,x2,x3,x4),dim=1)
        else:
           x=torch.cat((x1,x3,x4),dim=1)
        return self.bn(x)

class MixedInceptionModule(nn.Module):
    def __init__(self,mlength,channels):
        super().__init__()
        #self.conv1=nn.Conv2d(5,32,kernel_size=1,stride=1,padding=0)
        #self.mibs=nn.ModuleList()
        #self.mlength=mlength
        #self.channels=channels
        #self.norm=nn.BatchNorm2d(32)
        
        self.conv1=nn.Conv2d(5,64,kernel_size=1,stride=1,padding=0)
        self.mibs=nn.ModuleList()
        self.mlength=mlength
        self.channels=channels
        self.norm=nn.BatchNorm2d(64)
        j=0
        for i in range(mlength):
            if i==mlength-1:
               self.mibs.append(nn.Sequential(nn.AvgPool2d(kernel_size=2),
                                                  MixedInceptionModuleBlock(self.channels[j],self.channels[j+1],use_5x5_conv=False)))
            elif i%2==0:
               self.mibs.append(nn.Sequential(nn.AvgPool2d(kernel_size=2),MixedInceptionModuleBlock(self.channels[j],self.channels[j+1],use_5x5_conv=True)))
            else:
               self.mibs.append(MixedInceptionModuleBlock(self.channels[j],self.channels[j+1],use_5x5_conv=True))
            j=j+2
        #768
        self.fc=nn.Sequential(nn.BatchNorm1d(768),nn.Linear(768,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Linear(256,16),nn.ReLU(),nn.BatchNorm1d(16),nn.Linear(16,2))
                
    def forward(self,x):
        x=self.conv1(x)
        x=self.norm(x)
        for i in range(self.mlength):
            x=self.mibs[i](x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
def VGG_diy():
    #return MixedInceptionModule(5,[32,16,64,8,32,8,32,4,16,4])
    #return MixedInceptionModule(5,[32,16,64,8,24,8,32,4,16,4])
    return MixedInceptionModule(5,[64,32,128,64,256,16,64,4,16,4])

if __name__=='__main__':
    model=VGG_diy()
    #model=vit_demo()
    x=torch.rand((5,5,64,64))
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print(model(x).shape)
