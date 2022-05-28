import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Identity

class MaskedCNN(nn.Conv2d):
	def __init__(self, mask_type, *args, **kwargs):
		self.mask_type = mask_type
		assert mask_type in ['A', 'B'], "Unknown Mask Type"
		super(MaskedCNN, self).__init__(*args, **kwargs)
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0

	def forward(self, x):
		self.weight.data*=self.mask
		return super(MaskedCNN, self).forward(x)






class StackedLocalLayer(nn.Module):
    def __init__(self, x_channels, channels=128, local_size=3):
        super(StackedLocalLayer, self).__init__()
        self.local_size=local_size
        self.activation=nn.ReLU()

        self.first_layer=MaskedCNN('A',x_channels, channels, 3, 1, 1, bias=False)
        self.layers=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 3, 1, 1) for i in range(0,local_size-1)])
    
    def forward(self,x):
        x=self.first_layer(x)
        for i in range(self.local_size-1):
            x=self.activation(self.self.layers[i](x))
        
        return x




class LocalPixelCNN(nn.Module):
    def __init__(self, res_num=10, local_size = 3, x_channels=3,in_channels_z=10, channels=256, out_channels=100,if_bn=True, single_local_layer=False, device=None):
        super(LocalPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num
        
        if single_local_layer:
            self.locallayer=MaskedCNN('A',x_channels,channels//2, local_size*2+1, 1, local_size, bias=False)
        else:
            self.locallayer=StackedLocalLayer(x_channels,channels//2,local_size)
        
        
        self.in_cnn_2=MaskedCNN('B',in_channels_z,channels//2, 1, 1, 0, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
 
        self.out_cnn1=nn.Conv2d(channels, channels, 1)

        if if_bn:
            self.bn1=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn2=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn3=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn_out=nn.BatchNorm2d(channels)
        else:
            self.bn1=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn2=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn3=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn_out=Identity()

        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)
        
    def forward(self, x,z_out):
        x_1=self.locallayer(x)        
        x_2=self.in_cnn_2(z_out)
        x=torch.cat((x_1,x_2),1)
        x=self.activation(x)
        for i in range(0, self.res_num):
            x_mid=self.bn1[i](self.resnet_cnn11[i](x))
            x_mid=self.activation(x_mid)
            x_mid=self.bn2[i](self.resnet_cnn3[i](x_mid))
            x_mid=self.activation(x_mid)
            x_mid=self.bn3[i](self.resnet_cnn12[i](x_mid))
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.bn_out(self.out_cnn1(x))
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x
    

class FullPixelCNN(nn.Module): 
    def __init__(self, res_num=10, in_kernel = 7, kernel=7,  x_channels=3,in_channels_z=10, channels=256, out_channels=30,if_bn=True, device=None):
        super(FullPixelCNN, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num


        self.in_cnn_1=MaskedCNN('A',x_channels,channels//2, in_kernel, 1, in_kernel//2, bias=False)
        self.in_cnn_2=MaskedCNN('B',in_channels_z,channels//2, 1, 1, 0, bias=False)
        
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B', channels, channels, 3, 1, 1) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B', channels, channels, 3, 1, 1) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B', channels, channels, 3, 1, 1) for i in range(0,res_num)])
        
        if if_bn:
            self.bn1=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn2=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn3=torch.nn.ModuleList([nn.BatchNorm2d(channels) for i in range(0,res_num)])
            self.bn_out=nn.BatchNorm2d(channels)
        else:
            self.bn1=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn2=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn3=torch.nn.ModuleList([Identity() for i in range(0,res_num)])
            self.bn_out=Identity()


        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)

        
    def forward(self, x,z_out):
        x_1=self.in_cnn_1(x)        
        x_2=self.in_cnn_2(z_out)
        
        x=torch.cat((x_1,x_2),1)
        x=self.activation(x)

        for i in range(0, self.res_num):
            x_mid=self.bn1[i](self.resnet_cnn11[i](x))
            x_mid=self.activation(x_mid)
            x_mid=self.bn2[i](self.resnet_cnn3[i](x_mid))
            x_mid=self.activation(x_mid)
            x_mid=self.bn3[i](self.resnet_cnn12[i](x_mid))
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.bn_out(self.out_cnn1(x))
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x
    