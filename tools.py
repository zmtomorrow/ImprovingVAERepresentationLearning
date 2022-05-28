import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import torch.nn.functional as F


def batch_KL_diag_gaussian_std(mu_1,std_1,mu_2,std_2):
    diag_1=std_1**2
    diag_2=std_2**2
    ratio=diag_1/diag_2
    return 0.5*(torch.sum((mu_1-mu_2)**2/diag_2,dim=-1)+torch.sum(ratio,dim=-1)-torch.sum(torch.log(ratio),dim=-1)-mu_1.size(1))


def low_rank_cov_inverse(L,sigma):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    inner_inverse=torch.inverse(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)
    return inverse_var*torch.diag(torch.ones([dim]))-inverse_var**2*L@inner_inverse@L.t()

def low_rank_gaussian_logdet(L,sigma):
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    return torch.logdet(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)+dim*torch.log(var)


def KL_low_rank_gaussian_with_diag_gaussian(mu_1,L_1,sigma_1,mu_2,sigma_2,cuda):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    if cuda:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]).cuda())+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]).cuda())*var_1
    else:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1
    mu_diff=(mu_1-mu_2).view(-1,1)
    var_2=sigma_2**2
    return -0.5*(logdet_1-dim_1*torch.log(var_2)+dim_1-(1/var_2)*torch.trace(cov_1)-(1/var_2)*mu_diff.t()@mu_diff)



def KL_low_rank_gaussian_with_low_rank_gaussian(mu_1,L_1,sigma_1,mu_2,L_2,sigma_2):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
    cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1


    dim_2=L_2.size(0)
    rank_2=L_2.size(1)
    var_2=sigma_2**2
    inverse_var_2=1.0/var_2
    logdet_2=torch.logdet(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)+dim_1*torch.log(var_2)

    inner_inverse_2=torch.inverse(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)
    cov_inverse_2=inverse_var_2*torch.diag(torch.ones([dim_2]))-inverse_var_2**2*L_2@inner_inverse_2@L_2.t()

    mu_diff=(mu_1-mu_2).view(-1,1)
    return -0.5*(logdet_1-logdet_2+dim_1-torch.trace(cov_1@cov_inverse_2)-mu_diff.t()@ cov_inverse_2@mu_diff)


def general_kl_divergence(mu_1,cov_1,mu_2,cov_2):
    mu_diff=(mu_1-mu_2).view(-1,1)
    cov_2_inverse=torch.inverse(cov_2)
    return -0.5*(torch.logdet(cov_1)-torch.logdet(cov_2)+mu_1.size(0)-torch.trace(cov_1@cov_2_inverse)-mu_diff.t()@cov_2_inverse@mu_diff)

def low_rank_gaussian_one_sample(mu,L,sigma,cuda):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([rank]).cuda()
        eps=torch.randn([dim]).cuda()
    else:
        eps_z=torch.randn([rank])
        eps=torch.randn([dim])

    return eps_z@L.t()+eps*sigma+mu

def low_rank_gaussian_sample(mu,L,sigma,amount,cuda):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([amount,rank]).cuda()
        eps=torch.randn([amount,dim]).cuda()
    else:
        eps_z=torch.randn([amount,rank])
        eps=torch.randn([amount,dim])

    return eps_z@L.t()+eps*sigma+mu


def sample_from_batch_categorical(batch_logits,cuda):
    ### shape batch*dim
    ### gumbel max trick
    if cuda:
        noise = torch.rand(batch_logits.size()).cuda()
    else:
        noise = torch.rand(batch_logits.size())
    return torch.argmax(batch_logits - torch.log(-torch.log(noise)), dim=-1)


def sample_from_batch_categorical_multiple(batch_logits,sample_num,cuda):
    ### shape batch*dim
    ### gumbel max trick
    shape=list(batch_logits.size())
    shape.insert(-1, sample_num)
    if cuda:
        noise = torch.rand(shape).cuda()
    else:
        noise = torch.rand(shape)
    batch_logits_multiple=batch_logits.repeat(1,1,1,sample_num).view(shape)
    return torch.argmax(batch_logits_multiple - torch.log(-torch.log(noise)), dim=-1)


def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]



def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_ci_loss(x, l):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    nr_mix = int(ls[-1] / 7) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs),[1,2])

def discretized_mix_logistic_loss(x, l):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs),[1,2])


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs),(1,2))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.to(l.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.to(l.device)
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
       x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
       x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.to(l.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.to(l.device)
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def discretized_mix_logistic_uniform(x, l, alpha=0.0001):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
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
    return torch.sum(-log_probs,(1,2,3)) ### sum over three channels for test 