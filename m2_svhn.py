import torch
from main import *



opt={}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt['device']= torch.device('cuda:0')
    opt['if_cuda']=True
else:
    opt['device']= torch.device('cpu')
    opt['if_cuda']=False

opt['labels_per_class']=100
opt['data_set']='SVHN'
opt['dataset_path']='../data/svhn'
opt['save_path']='./save/'
opt['epochs'] = 20
opt['batch_size'] = 32
opt['validation_batch_size']=100
opt['alpha_coef']=0.1
opt['lr']=1e-3
opt['if_original_m2']=False
opt['if_local']=True
opt['z_dim']=64
opt['local_kernel_size']=5
opt['class_bn']=True
opt['dropout']=0.3


def get_name(opt):
    return opt['data_set']+'_lpc'+str(opt['labels_per_class'])+'ls'+str(opt['local_kernel_size'])+'zdim'+str(opt['z_dim'])+'_dropout'+str(opt['dropout'])+'_seed'+str(seed)


for seed in range(0,5):
    opt['seed']=seed
    print("--------------------------   Start seed: ", seed, "   ------------------------")
    test_accuracy=main(opt)
    np.save(opt['save_path']+get_name(opt),test_accuracy)







    
