import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import *
from pixelcnn import *
import numpy as np
from classification import *
from tools import *
from utils import *
import math
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from distributions import *



class PixelVAE(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.z_dim=opt['z_dim']
        self.device=opt['device']
        self.decoder_channels=opt['decoder_channels']
        self.prior=opt['prior']

        if opt['data_set'] in ['CIFAR','SVHN']:
            self.x_channels=3
            if opt['x_dis']=='logistic_ca':
                self.out_channels=9
                self.criterion=lambda data,params : -batch_logistic_autoregressive_logp(params[:,0:3,:,:],params[:,3:6,:,:],params[:,6:9,:,:],data)
            elif opt['x_dis']=='mix_logistic_ca':
                self.out_channels=100
                self.criterion=lambda real, fake : discretized_mix_logistic_loss(real, fake)
            elif opt['x_dis']=='mix_logistic_ci':
                self.out_channels=70
                self.criterion=lambda real, fake : discretized_mix_logistic_ci_loss(real, fake)
            if opt['net']=='dc':
                self.encoder=dc_encoder(z_dim=self.z_dim,if_bn=opt['if_ed_bn'])
                self.decoder=dc_decoder(z_dim=self.z_dim,out_channels=self.decoder_channels,if_bn=opt['if_ed_bn'])
            if opt['net']=='res':
                self.encoder=res_encoder(z_dim=self.z_dim,res_num=3)
                self.decoder=res_decoder(z_dim=self.z_dim,out_channels=self.decoder_channels)
        else:
            self.x_channels=1
            self.encoder=densenet_encoder(z_dim=self.z_dim,if_bn=opt['if_ed_bn'])
            self.decoder=densenet_decoder(o_dim=self.decoder_channels,z_dim=self.z_dim,if_bn=opt['if_ed_bn'])
            if opt['data_set']=='BinaryMNIST':
                self.out_channels=1
                self.criterion  = lambda  real,fake : -Bernoulli(logits=fake).log_prob(real).sum([1,2,3])
            elif opt['data_set']=='MNIST':
                self.out_channels=30
                self.criterion  = lambda  real,fake :discretized_mix_logistic_loss_1d(real, fake)

        if opt['if_local']:
            self.pixelcnn=LocalPixelCNN(res_num=5, local_size = opt['local_size'], x_channels=self.x_channels,in_channels_z=self.decoder_channels, out_channels=self.out_channels,if_bn=opt['if_pixel_bn'],single_local_layer=opt['single_local_layer'])
        else:
            self.pixelcnn=FullPixelCNN(res_num=5, in_kernel = 7,  x_channels=self.x_channels,in_channels_z=self.decoder_channels,  out_channels=self.out_channels,if_bn=opt['if_pixel_bn'])
        
        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())
        self.optimizer = optim.Adam(self.params, lr=opt['lr'])

    
    def forward(self,x):
        z_mu, z_std=self.encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        z=eps.mul(z_std).add_(z_mu)
        vae_out=self.decoder(z)
        if self.prior=='Gaussian':
            kl=batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        elif self.prior=='Laplace':
            logqz_x=Normal(z_mu,z_std).log_prob(z).sum(-1)
            logpz=Laplace(self.prior_mu.to(self.device),self.prior_std.to(self.device)).log_prob(z).sum(-1)
            kl=logqz_x-logpz
        else:
            raise NotImplementedError

        output = self.pixelcnn(x,vae_out)
        neg_l = self.criterion(x, output)
        loss = torch.mean(neg_l+kl,dim=0)
        return loss,torch.mean(kl)


class VAE(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.z_dim=opt['z_dim']
        if opt['data_set'] in ['CIFAR','SVHN']:
            self.encoder=dc_encoder(z_dim=self.z_dim)
            self.decoder=dc_decoder(z_dim=self.z_dim, out_channels=3, h_dim=256)
            self.criterion=lambda real, fake : discretized_mix_logistic_loss(real, fake)
        else:
            self.encoder=densenet_encoder( z_dim=self.z_dim)
            self.decoder=densenet_decoder(o_dim=10,z_dim=self.z_dim)
            self.criterion  = lambda  real,fake :discretized_mix_logistic_loss_1d(real, fake)
        self.device=opt['device']
        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())
        self.optimizer = optim.Adam(self.params, lr=opt['lr'])

    
    def forward(self,x):
        z_mu, z_std=self.encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        z=eps.mul(z_std).add_(z_mu)
        vae_out=self.decoder(z)
        kl=batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        neg_l = self.criterion(x, vae_out)
        loss = torch.mean(neg_l+kl,dim=0)
        return loss    


class M2(nn.Module):
    def __init__(self,  opt):
        super().__init__()
        self.z_dim=opt['z_dim']
        self.device=opt['device']
        self.if_original_m2=opt['if_original_m2']
        self.if_local=opt['if_local']

        self.y_dim=10
        if opt['data_set'] in ['CIFAR','SVHN']:
            self.decoder_channels=100
            self.x_channels=3
            self.out_channels=100
            if self.if_original_m2:
                self.decoder_channels=self.out_channels
            self.decoder=dc_decoder(z_dim=self.z_dim, out_channels=self.decoder_channels, if_label=True)
            self.encoder=dc_encoder(z_dim=self.z_dim, if_label=True)
            self.criterion=lambda real, fake : discretized_mix_logistic_loss(real, fake)
            self.classify = m2_classifier_color(if_bn=opt['class_bn'],drop_out_rate=opt['dropout'])

        else:
            self.x_flat_dim=784
            self.x_channels=1
            self.decoder_channels=32
            if opt['data_set']=='BinaryMNIST':
                self.out_channels=1
                self.criterion  = lambda  real,fake : -Bernoulli(logits=fake).log_prob(real).sum([1,2,3])
            elif opt['data_set']=='MNIST':
                self.out_channels=30
                self.criterion  = lambda  real,fake :discretized_mix_logistic_loss_1d(real, fake)

            self.en_input_dim = self.x_flat_dim + self.y_dim
            self.de_input_dim = self.y_dim + self.z_dim
            if self.if_original_m2:
                self.decoder_channels=self.out_channels
            self.encoder=densenet_encoder(input_dim=self.en_input_dim, z_dim=self.z_dim)
            self.decoder=densenet_decoder(o_dim=self.decoder_channels, z_dim=self.de_input_dim)
            self.classify = m2_classifier(if_bn=opt['class_bn'])


        if self.if_local and not self.if_original_m2:
            self.pixelcnn=LocalPixelCNN(res_num=5, local_size = opt['local_size'], x_channels=self.x_channels,in_channels_z=self.decoder_channels, out_channels=self.out_channels,if_bn=False,single_local_layer=opt['single_local_layer'])
        elif not self.if_original_m2:
            self.pixelcnn=FullPixelCNN(res_num=5, in_kernel = 7,  x_channels=self.x_channels,in_channels_z=self.decoder_channels,  out_channels=self.out_channels,if_bn=False)
        
        self.device=opt['device']
        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())

    def generative_classifier(self,xs):
        batch_size = xs.size(0)
        ys = torch.from_numpy(np.arange(self.y_dim))
        ys = ys.view(-1,1).repeat(1, batch_size).view(-1)
        ys = one_hot(ys, self.y_dim)
        ys = Variable(ys.float())
        ys = ys.to(self.device) if xs.is_cuda else ys
        xs = xs.repeat(self.y_dim, 1, 1, 1)
        z_mu, z_std = self.encoder(xs, ys)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        if self.if_original_m2:
            x_recon = self.decoder(zs, ys)
        else:
            vae_out = self.decoder(zs, ys)
            x_recon = self.pixelcnn(xs, vae_out)
        loglikelihood = -self.criterion(xs, x_recon)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        c_likelihood=torch.transpose((loglikelihood-kl-math.log(self.y_dim)).view(self.y_dim,batch_size),0,1)
        return F.softmax(c_likelihood, dim=-1)



    # Calculating ELBO (Both labelled or unlabelled)
    def forward(self, x, y=None):
        labelled = False if y is None else True

        xs, ys = (x, y)
        # Duplicating samples and generate labels if not labelled
        if not labelled:
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(self.y_dim))
            ys = ys.view(-1,1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, self.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(self.device) if xs.is_cuda else ys
            xs = xs.repeat(self.y_dim, 1, 1, 1)
        
        # Reconstruction
        z_mu, z_std = self.encoder(xs, ys)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        if self.if_original_m2:
            x_recon = self.decoder(zs, ys)
        else:
            vae_out = self.decoder(zs, ys)
            x_recon = self.pixelcnn(xs, vae_out)

        # p(x|y,z)
        
        loglikelihood = -self.criterion(xs, x_recon)

        # p(y)
        logprior_y = -math.log(self.y_dim)

        # KL(q(z|x,y)||p(z))
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))

        # ELBO : -L(x,y)
        neg_L = loglikelihood + logprior_y - kl

        if labelled:
            return torch.mean(neg_L)

        prob_y = self.classify(x)

        neg_L = neg_L.view_as(prob_y.t()).t()

        # H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(prob_y, torch.log(prob_y + 1e-8)), dim=-1)
        neg_L = torch.sum(torch.mul(prob_y, neg_L), dim=-1)

        # ELBO : -U(x)
        neg_U = neg_L + H
        return torch.mean(neg_U)









    
    
    
    
