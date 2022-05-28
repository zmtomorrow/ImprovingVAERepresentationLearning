import torch
from torch import optim
from utils import *
from model import *
import numpy as np
import torch.nn.functional as F
from classification import *
from itertools import cycle
from tqdm import tqdm
from torch.distributions import Normal

def PCA(model,test,opt):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=opt['z_dim'],whiten=True)

    model.eval()
    r_list=[]
    with torch.no_grad():
        for x, y in test:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)

            z_mu,z_std=model.encoder(x.to(opt['device']))
            if opt['sto_rep']:
                z=torch.randn_like(z_mu)*z_std+z_mu
            else:
                z=z_mu
            r_list.append(z.cpu().detach().numpy())

    pca.fit(np.asarray(r_list).reshape(-1,opt['z_dim']))
    return pca.explained_variance_ratio_


def LinerSVM(model,train,test,opt):
    from sklearn.svm import SVC
    model.eval()
    clf=SVC(kernel='linear')
    r_list=[]
    label_list=[]
    with torch.no_grad():
        for x, y in train:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)

            z_mu,z_std=model.encoder(x.to(opt['device']))
            if opt['sto_rep']:
                z=torch.randn_like(z_mu)*z_std+z_mu
            else:
                z=z_mu
            r_list.append(z.cpu().detach())
            label_list.append(y.numpy())
    clf.fit(torch.stack(r_list).view(-1,opt['z_dim']).numpy(),np.asarray(label_list).reshape(-1))

    r_list=[]
    label_list=[]
    with torch.no_grad():
        for x, y in test:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x).to(opt['device'])

            z_mu,z_std=model.encoder(x.to(opt['device']))
            if opt['sto_rep']:
                z=torch.randn_like(z_mu)*z_std+z_mu
            else:
                z=z_mu
            r_list.append(z.cpu().detach())
            label_list.append(y.numpy())
    accuracy=clf.score(torch.stack(r_list).view(-1,opt['z_dim']).numpy(), np.asarray(label_list).reshape(-1))

    return accuracy


def ConditionalEntropy(model,test,opt):
    with torch.no_grad():
        averge_entropy=0.
        for x, _ in test:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)

            z_mu,z_std=model.encoder(x.to(opt['device']))
            averge_entropy+=Normal(z_mu,z_std).entropy().sum(-1).mean()
    
    return averge_entropy/len(test)



def DetClassEval(model,train,test,opt):
    classifier=ClassNet(device=opt['device'],z_dim=opt['z_dim']).to(opt['device'])
    r_list=[]
    model.eval()
    with torch.no_grad():
        for x, y in train:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)

            z_mu,z_std=model.encoder(x.to(opt['device']))
            if opt['sto_rep']:
                z=torch.randn_like(z_mu)*z_std+z_mu
            else:
                z=z_mu
            r_list.append((z,y))
    classifier.train()
    for _ in range(opt['class_epochs']):
        for z,y in r_list:
            classifier.optimizer.zero_grad()
            logprobs=classifier.forward(z)
            loss=F.nll_loss(logprobs, y.to(opt['device']),reduction='mean')
            loss.backward()
            classifier.optimizer.step()
    classifier.eval()
    with torch.no_grad():
        accuracy=0.0
        for x, y in test:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)

            z_mu,z_std=model.encoder(x.to(opt['device']))
            if opt['sto_rep']:
                z=Normal(z_mu,z_std).sample([opt['sample_num']])
                pred=classifier.forward(z).mean(0).argmax(dim=-1).cpu()
            else:
                z=z_mu
                pred=classifier.predict(z).cpu()
            accuracy += torch.mean((pred == y).float()).item()
        accuracy=accuracy/len(test)
    return accuracy




def BPDEval(model,dataloader,opt):
    with torch.no_grad():
        model.eval()
        eval_BPD=0.
        eval_kl=0.
        for x, _ in dataloader:
            if opt['x_dis'][0:12]== 'mix_logistic':
                x = rescaling(x)
            loss,kl=model(x.to(opt['device']))
            eval_kl+=kl.item()
            eval_BPD+=loss.item()/np.log(2.0)
        return eval_BPD/(len(dataloader)*np.prod(x.size()[-3:])),eval_kl/len(dataloader)


def UnsupervisedTrain(model,dataloader,opt):
    model.train()
    for x, _ in dataloader:
        if opt['x_dis'][0:12]== 'mix_logistic':
            x = rescaling(x)

        model.optimizer.zero_grad()
        loss,_ = model(x.to(opt['device']))
        loss.backward()
        model.optimizer.step()
    return (loss/np.log(2.0)).item()



def SemiTrain(model,labelled,unlabelled,optimizer,epoch,opt):
    model.train()
    total_loss, accuracy = (0, 0)
    model.classify.train()
    assert len(unlabelled) > len(labelled)
    with tqdm(unlabelled, unit="batch") as unlabelled_tqdm:
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled_tqdm):
            unlabelled_tqdm.set_description(f"Epoch {epoch}")
            current_batch_size = u.size()[0]
            if opt['data_set']=='BinaryMNIST':
                x = x.to(opt['device'])
                u = u.to(opt['device'])
            else:
                x = rescaling(x).to(opt['device'])
                u = rescaling(u).to(opt['device'])
            y = one_hot(y, num_classes=10).to(opt['device'])
            optimizer.zero_grad()


            L = -model(x, y)
            U = -model(u)

            # Add auxiliary classification loss q(y|x)
            prob_y = model.classify(x)
            
            # cross entropy
            classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()

            J_alpha = L + U + opt['alpha'] * classication_loss

            J_alpha.backward()
            optimizer.step()
            
            total_loss += J_alpha.item()
            accuracy += torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

            batch_loss = J_alpha.item() / current_batch_size
            batch_acc = torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item() / current_batch_size
            unlabelled_tqdm.set_postfix(loss=batch_loss, accuracy=100. * batch_acc)

    m = len(unlabelled)
    print("Epoch: {}".format(epoch))
    print("[Train]\t\t J_a: {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def SemiTest(model,validation,opt):
    model.eval()
    model.classify.eval()
    total_loss, accuracy = (0, 0)
    generative_accuracy=0
    with torch.no_grad():
        for x, y in validation:
            if opt['data_set']=='BinaryMNIST':
                x = x.to(opt['device'])
            else:
                x = rescaling(x).to(opt['device'])
            y = one_hot(y, num_classes=10).to(opt['device'])
           
            L = -model(x, y)
            U = -model(x)
            prob_y = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()

            J_alpha = L + U + opt['alpha'] * classication_loss

            total_loss += J_alpha.item()

            _, pred_idx = torch.max(prob_y, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx == lab_idx).float()).item()

            if opt['gen_classifier']==True:
                generative_prob_y=model.generative_classifier(x)
                _, gen_pred_idx = torch.max(generative_prob_y, 1)
                generative_accuracy+=torch.mean((gen_pred_idx == lab_idx).float()).item()
        
        m = len(validation)
        print("[Validation]\t J_a: {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
        if opt['gen_classifier']==True:
            print('generative_classifier:', generative_accuracy/m)

        return total_loss / m, accuracy / m


def SemiMain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    labelled,unlabelled,validation=LoadSemiData(opt)
    
    opt['alpha'] = opt['alpha_coef'] * (len(unlabelled)+len(labelled)) / len(labelled)

    model=M2(opt).to(opt['device'])

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
    # sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, 10) 

    train_loss=[]
    test_loss=[]

    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(1, opt['epochs'] + 1):
        train_l, train_acc = SemiTrain(model,labelled,unlabelled,optimizer,epoch,opt)
        test_l, test_acc = SemiTest(model,validation,opt)
        train_loss.append(train_l)

        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    return test_accuracy




    