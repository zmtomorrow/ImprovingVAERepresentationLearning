import torch
from utils import *
from model import *
from main import *


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, required=True)
parser.add_argument('--mode', type=int, required=True)

dataset= parser.parse_args().data
mode= parser.parse_args().mode


print(dataset)

print(mode)


opt = {}
opt=get_device(opt,gpu_index=str(mode%2))
opt['data_set']=dataset
opt['dataset_path']='../data/'
opt['save_path']='./new_save/'
opt['result_path']='./new_results/'
opt['epochs'] = 500
opt['class_epochs']=100
opt['batch_size'] = 100
opt['test_batch_size']=1000
opt['lr']=3e-4
opt['seed']=0
opt['if_ed_bn']=True
opt['if_pixel_bn']=False
opt['sto_rep']=False

if dataset in ['CIFAR','SVHN']:
    opt['z_dim']=64
    opt['decoder_channels']=256
    opt['net']='res'
    opt['data_aug']=True

    if mode==0:
        opt['if_local']=True
        opt['local_kernel_size']=1
        opt['x_dis']='mix_logistic_ci'
    elif mode==1:
        opt['if_local']=True
        opt['local_kernel_size']=1
        opt['x_dis']='mix_logistic_ca'
    elif mode==2:
        opt['if_local']=True
        opt['local_kernel_size']=3
        opt['x_dis']='mix_logistic_ca'
    elif mode==3:
        opt['if_local']=True
        opt['local_kernel_size']=5
        opt['x_dis']='mix_logistic_ca'
    elif mode==4:
        opt['if_local']=False
        opt['x_dis']='mix_logistic_ca'

elif dataset=='MNIST':
    opt['z_dim']=32
    opt['decoder_channels']=16
    opt['epochs'] = 50
    if mode==0:
        opt['if_local']=True
        opt['x_dis']='mix_logistic_ci'
        opt['local_kernel_size']=1
    elif mode==1:
        opt['if_local']=True
        opt['x_dis']='mix_logistic_ci'
        opt['local_kernel_size']=3
    elif mode==2:
        opt['if_local']=True
        opt['x_dis']='mix_logistic_ci'
        opt['local_kernel_size']=5
    elif mode==3:
        opt['if_local']=False
        opt['x_dis']='mix_logistic_ci'


np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])
train_data_loader,test_data_loader,train_data_evaluation=LoadData(opt)

net=PixelVAE(opt).to(opt['device'])

train_BPD_list=[]
test_BPD_list=[]
train_kl_list=[]
test_kl_list=[]
test_det_acc_list=[]
linear_acc_list=[]

name=opt['data_set']+'_mode'+str(mode)
if opt['if_local']:
    name+='_ls'+str(opt['local_kernel_size'])

for e in range(0,opt['epochs']+1):
    print('epoch',e)
    UnsupervisedTrain(net,train_data_loader,opt)    
    train_bpd,train_kl=BPDEval(net,train_data_evaluation,opt)
    test_bpd,test_kl=BPDEval(net,test_data_loader,opt)
    test_BPD_list.append(test_bpd)
    train_BPD_list.append(train_bpd)
    train_kl_list.append(train_kl)
    test_kl_list.append(test_kl)
    print('train BPD',train_bpd)
    print('test BPD',test_bpd)
    print('train kl',train_kl)
    print('test kl',test_kl)

    if e%10==0:
        det_accuracy=DetClassEval(net,train_data_loader,test_data_loader,opt)
        linear_acc=LinerSVM(net,train_data_loader,test_data_loader,opt)
        test_det_acc_list.append(det_accuracy)
        linear_acc_list.append(linear_acc)
        print('test det acc',det_accuracy)
        print('test linear acc',linear_acc)

        np.save(opt['result_path']+name+'_trainkl',train_kl_list)
        np.save(opt['result_path']+name+'_testkl',test_kl_list)

        np.save(opt['result_path']+name+'_trainBPD',train_BPD_list)
        np.save(opt['result_path']+name+'_testBPD',test_BPD_list)
        np.save(opt['result_path']+name+'_testDetAcc',test_det_acc_list)
        np.save(opt['result_path']+name+'_testLinearAcc',linear_acc_list)

        if e%10==0:
            torch.save(net.state_dict(),opt['save_path']+name+'.pth')
