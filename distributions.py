import torch
import numpy as np
from tools import *


def batch_logistic_logp(means, log_scales, x):
    """
    mean: [n,c,h,w]
    log_scale: [n,c,h,w]
    x: [n,c,h,w]
    """
    centered_x = x - means
    inv_stdv = torch.exp(-torch.clamp(log_scales, min=-7.0))
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1.0 / 255.0))
    cdf_min = torch.sigmoid(inv_stdv * (centered_x))
    cdf_plus = torch.where(x > 0.999, torch.ones(1).to(x.device), cdf_plus)
    cdf_min = torch.where(x < 0.001, torch.zeros(1).to(x.device), cdf_min)
    return torch.sum(torch.log(cdf_plus - cdf_min + 1e-7), (1, 2, 3)).mean()



def batch_logistic_sample(means, log_scales):
    """
    mean: [n,c,h,w]
    log_scale: [n,c,h,w]
    x: [n,c,h,w]
    """
    std = torch.exp(torch.clamp(log_scales, min=-7.0))
    u = torch.rand_like(means)
    continous_sample = torch.clamp(
        means + std * (torch.log(u) - torch.log(1 - u)), 0.0, 1.0
    )
    return torch.round(continous_sample * 255.0) / 255.0


def batch_logistic_autoregressive_logp(means,log_scales,mean_linear,x):
    """
        channel-wise autoregressive with linear condition
        means: [n,3,h,w]
        log_scale: [n,3,h,w]
        x: [n,3,h,w]
        mean_linear: [n,3,h,w]
    """
    # mean_linear=torch.sigmod(mean_linear)
    m2=means[:,1:2,:,:]+x[:,0:1,:,:]*mean_linear[:,0:1,:,:]
    m3=means[:,2:3,:,:]+x[:,0:1,:,:]*mean_linear[:,1:2,:,:]+mean_linear[:,2:3,:,:]*x[:,1:2,:,:]
    means=torch.cat((means[:,0:1,:,:],m2,m3),dim=1)                                                                                                  
    centered_x = x - means                                                                                                                           
    inv_stdv = torch.exp(-torch.clamp(log_scales,min=-7.))                                                                                        
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))
    cdf_min = torch.sigmoid(inv_stdv * centered_x )
    cdf_plus=torch.where(x > 0.999, torch.ones(1).to(x.device),cdf_plus)
    cdf_min=torch.where(x < 0.001, torch.zeros(1).to(x.device),cdf_min)
    return torch.sum(torch.log(cdf_plus-cdf_min+1e-7),(1,2,3)).mean()



def batch_logistic_mean_transoform_logp(means,log_scales,mean_linear,x):
    """
        channel-wise autoregressive with linear condition
        means: [n,3,h,w]
        log_scale: [n,3,h,w]
        x: [n,3,h,w]
        mean_linear: [n,3,h,w]
    """
    # mean_linear=torch.sigmod(mean_linear)
    m2=means[:,1:2,:,:]+means[:,0:1,:,:]*mean_linear[:,0:1,:,:]
    m3=means[:,2:3,:,:]+means[:,0:1,:,:]*mean_linear[:,1:2,:,:]+mean_linear[:,2:3,:,:]*means[:,1:2,:,:]
    means=torch.cat((means[:,0:1,:,:],m2,m3),dim=1)                                                                                                  
    centered_x = x - means                                                                                                                           
    inv_stdv = torch.exp(-torch.clamp(log_scales,min=-7.))                                                                                        
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))
    cdf_min = torch.sigmoid(inv_stdv * centered_x )
    cdf_plus=torch.where(x > 0.999, torch.ones(1).to(x.device),cdf_plus)
    cdf_min=torch.where(x < 0.001, torch.zeros(1).to(x.device),cdf_min)
    return torch.sum(torch.log(cdf_plus-cdf_min+1e-7),(1,2,3)).mean()


def batch_logistic_mean_transoform_sample(means,log_scales,mean_linear):
    """
        channel-wise autoregressive with linear condition
        means: [n,3,h,w]
        log_scale: [n,3,h,w]
        x: [n,3,h,w]
        mean_linear: [n,3,h,w]
    """
    # mean_linear=torch.sigmod(mean_linear)
    m2=means[:,1:2,:,:]+means[:,0:1,:,:]*mean_linear[:,0:1,:,:]
    m3=means[:,2:3,:,:]+means[:,0:1,:,:]*mean_linear[:,1:2,:,:]+mean_linear[:,2:3,:,:]*means[:,1:2,:,:]
    means=torch.cat((means[:,0:1,:,:],m2,m3),dim=1)    
    std = torch.exp(torch.clamp(log_scales, min=-7.0))
    u = torch.rand_like(means)                                                                                              
    continous_sample = torch.clamp(
        means + std * (torch.log(u) - torch.log(1 - u)), 0.0, 1.0
    )
    return torch.round(continous_sample * 255.0) / 255.0



def discretized_mix_logistic_uniform_logp(x, l, alpha=0.0001):
    x = x.permute(0, 2, 3, 1)   
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
   
    nr_mix = int(l.size(-1) / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous().unsqueeze(-1)
    m2 = means[:, :, :,1:2, :] + coeffs[:, :, :, 0:1, :]* x[:, :, :, 0:1, :]

    m3 = means[:, :, :, 2:3, :] + coeffs[:, :, :, 1:2, :] * x[:, :, :, 0:1, :] +coeffs[:, :, :, 2:3, :] * x[:, :, :, 1:2, :]

    means = torch.cat((means[:, :, :, 0:1, :], m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min=torch.where(x <- 0.999, torch.tensor(0.0).to(x.device),cdf_min)

    uniform_cdf_min = ((x+1.)/2*255)/256.
    uniform_cdf_plus = ((x+1.)/2*255+1)/256.

    pi=torch.exp(log_prob_from_logits(logit_probs)).unsqueeze(-2).repeat(1,1,1,3,1)

    mix_cdf_plus=((1-alpha)*pi*cdf_plus+(alpha/nr_mix)*uniform_cdf_plus).sum(-1)
    mix_cdf_min=((1-alpha)*pi*cdf_min+(alpha/nr_mix)*uniform_cdf_min).sum(-1)

    log_probs =torch.log(mix_cdf_plus-mix_cdf_min)
    return -log_probs.sum(1,2,3) 




def discretized_mix_logistic_logp(x, l):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
   
    nr_mix = int(l.size(-1) / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous().unsqueeze(-1)
    m2 = means[:, :, :,1:2, :] + coeffs[:, :, :, 0:1, :]* x[:, :, :, 0:1, :]

    m3 = means[:, :, :, 2:3, :] + coeffs[:, :, :, 1:2, :] * x[:, :, :, 0:1, :] +coeffs[:, :, :, 2:3, :] * x[:, :, :, 1:2, :]

    means = torch.cat((means[:, :, :, 0:1, :], m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min=torch.where(x <- 0.999, torch.tensor(0.0).to(x.device),cdf_min)
    pi=torch.exp(log_prob_from_logits(logit_probs)).unsqueeze(-2).repeat(1,1,1,3,1)
    mix_cdf_plus=(pi*cdf_plus).sum(-1)
    mix_cdf_min=(pi*cdf_min).sum(-1)
    log_probs =torch.log(mix_cdf_plus-mix_cdf_min+1e-7)
    return torch.sum(log_probs,(1,2,3))


def discretized_mix_logistic_sample(l, nr_mix):
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.to(l.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.to(l.device)
    u.uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
       x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
       x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)
    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    out = out.permute(0, 3, 1, 2)
    return ((.5 * out  + .5)*255).int()/255.




if __name__ == "__main__":
    l=torch.randn(100,100,32,32)
    x=torch.randn(100,3,32,32)
    # print(discretized_mix_logistic_sample(l,10))
    # print(discretized_mix_logistic_logp_o(x, l))
