import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from operator import __or__
import torch
import torchvision
from torch.utils import data
import torch.nn as nn
from torchvision import transforms



rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x


def get_device(opt,gpu_index):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        opt["device"] = torch.device("cuda:"+str(gpu_index))
        opt["if_cuda"] = True
    else:
        opt["device"] = torch.device("cpu")
        opt["if_cuda"] = False
    return opt


def plot_latent(autoencoder, data, device, num_batches=500):
    for i, (x, y) in enumerate(data):
        z,_ = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.plot()
    plt.show()
    
def gray_show_many(image,number_sqrt):
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
    
def color_show_many(image,number_sqrt,dim=32, channels=3):
    image=image.view(-1,3,32,32).permute(0,2,3,1)
    canvas_recon = np.empty((dim * number_sqrt, dim * number_sqrt, channels))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim,:] = \
            image[count]
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon)
    plt.show()


def get_sampler(labels, n=None):
    # Choose digits in 0-9 
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(10)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(10)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler
    


    
def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())
        
    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug']==True:
            trans=transforms.Compose([
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor()])
        else:
            trans=torchvision.transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader,test_data_loader,train_data_evaluation



def LoadSemiData(opt):
    if opt['data_set'] == 'SVHN':
        if opt['data_aug']==True:
            trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        else:
            trans = transforms.ToTensor()
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=trans)
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug']==True:
            trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        else:
            trans = transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    

    labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                            sampler=get_sampler(train_data.train_labels.numpy(),  opt['labels_per_class']))
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                                sampler=get_sampler(train_data.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                sampler=get_sampler(test_data.test_labels.numpy()))

    return labelled,unlabelled,validation
