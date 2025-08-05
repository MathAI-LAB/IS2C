# -*- coding: utf-8 -*-

##################
import torch
torch.cuda.empty_cache()
torch.cuda.set_device(0)
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy.io as io 
import h5py
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import os
import argparse
parser = argparse.ArgumentParser(description="IS2C_PDA")


parser.add_argument('--dataset', type=str, default='Office31')
args = parser.parse_args()
dset = args.dataset

# Training settings
#torch.manual_seed(0) 

#====== Hyper-Parameter ======
if dset=='VisDA':
    epochs = 5
else:
    epochs = 200
Exptimes = 10
if dset == 'ImageCLEF':
    FD_step =[1,5,10]
    pre = [0,1,5]
if dset == 'Office31':
    FD_step =[10,15,20]
    pre = [0,5,10]
else:
    FD_step =[5,10,15,20]
    pre = [0,5,10,15]
batch_size=5000
eps = 1
weight_decay=0.01
if dset=='VisDA' or dset=='OfficeHome':
    lr = 0.001
else: 
    lr = 0.0015#0.0003#0.001#0.0006
beta1 = 0.9
beta2 = 0.999

if dset=='Office31':
    FD_lam_list= [75, 100]
else:
    FD_lam_list= [75, 100, 125, 150, 175, 200, 225, 250]
alpha= 0.2#[0.05,0.1,0.25,0.5,0.75,0.9,0.95]#[0.1, 0.2, 0.4, 2, 4, 8]


#====== ImageCLEF ======
if dset == 'ImageCLEF':
    source_domain_set = ('I','P','I','C','C','P')#('C',)#
    target_domain_set = ('P','I','C','I','P','C')#('P',)#
    save_name = ('ItoP','PtoI','ItoC','CtoI','CtoP','PtoC')#('CtoP',)#('PtoC',)#
    partial_cls_idx = [0,1,2,3,4,5]
    n_classes = 12


#====== OfficeHome ======
elif dset == 'OfficeHome':
    source_domain_set = ('Art','Art','Art','Clipart','Clipart','Clipart','Product','Product','Product','Real_World','Real_World','Real_World')#('Clipart','Clipart','Clipart',)#('Art','Art','Art','Clipart','Clipart','Clipart','Product','Product','Product','Real_World','Real_World','Real_World')#
    #('Art','Clipart','Product',)#('Art','Clipart','Clipart','Clipart','Product',)#('Art','Clipart','Clipart','Clipart','Product','Real_World',)#
    target_domain_set = ('Clipart','Product','Real_World','Art','Product','Real_World','Art','Clipart','Real_World','Art','Clipart','Product')#('Art','Product','Real_World',)#('Clipart','Product','Real_World','Art','Product','Real_World','Art','Clipart','Real_World','Art','Clipart','Product')#
    #('Clipart','Real_World','Clipart',)#('Clipart','Art','Product','Real_World','Clipart',)#('Clipart','Art','Product','Real_World','Clipart','Clipart',)#
    save_name = ('ARtoCL','ARtoPR','ARtoRW','CLtoAR','CLtoPR','CLtoRW','PRtoAR','PRtoCL','PRtoRW','RWtoAR','RWtoCL','RWtoPR',)#('CLtoAR','CLtoPR','CLtoRW',)#
    #('ARtoCL','CLtoAR','PRtoCL',)#('ARtoCL','CLtoAR','CLtoPR','CLtoRW','PRtoCL',)#('ARtoCL','CLtoAR','CLtoPR','CLtoRW','PRtoCL','RWtoCL',)#
    partial_cls_idx = list(range(25))
    n_classes = 65

# ('Art',)#('Clipart',)#('Real_World',)#('Product',)#
# ('ARtoCL',)#('ARtoPR',)#('ARtoRW',)#('CLtoAR',)#('CLtoPR',)#('CLtoRW',)#('PRtoAR',)#('PRtoCL',)#
# ('PRtoRW',)#('RWtoAR',)#('RWtoCL',)#('RWtoPR',)#


#====== Office31 TIDOT ====
# dset = 'Office31TIDOT'
# n_classes = 31
# source_domain_set = ('amazon','amazon','dslr','dslr','webcam','webcam')
# target_domain_set = ('dslr','webcam','amazon','webcam','amazon','dslr')
# save_name = ('AtoD','AtoW','DtoA','DtoW','WtoA','WtoD')
# partial_cls_idx = [0,1,5,10,11,12,15,16,17,22]

#====== Office31 ======
elif dset == 'Office31':
    n_classes = 31
    source_domain_set = ('amazon','amazon','webcam','webcam','dslr','dslr')#('webcam',)#('amazon',)#
    target_domain_set = ('webcam','dslr','amazon','dslr','amazon','webcam')#('amazon',)#('dslr',)#
    save_name = ('AtoW','AtoD','WtoA','WtoD','DtoA','DtoW')#('WtoA',)#('AtoD',)#('DtoA',)#('AtoW',)#
    partial_cls_idx = [0,1,5,10,11,12,15,16,17,22]
#---------------------------------- VisDA --------------------------------

elif dset == 'VisDA':
    source_domain_set = ('synthetic',)
    target_domain_set = ('realistic',)
    save_name = ('StoR',)
    n_classes = 12
    partial_cls_idx = [0,1,2,3,4,5,]

##################################
# Load data and create Data loaders
##################################
def get_mat_path(dset,domain):
    data_dir = './DATA'
    if dset == 'ImageCLEF':
        dset_dir = 'ImageCLEF'
        if domain == 'I':
            file_name = 'imageclef-i-resnet50-noft.mat'
        elif domain == 'P':
            file_name = 'imageclef-p-resnet50-noft.mat'
        elif domain == 'C':
            file_name = 'imageclef-c-resnet50-noft.mat'
    elif dset == 'OfficeHome':
        dset_dir = 'OfficeHome'
        if domain == 'Art':
            file_name = 'OfficeHome-Art-resnet50-noft.mat'
        elif domain == 'Clipart':
            file_name = 'OfficeHome-Clipart-resnet50-noft.mat'
        elif domain == 'Product':
            file_name = 'OfficeHome-Product-resnet50-noft.mat'
        elif domain == 'Real_World':
            file_name = 'OfficeHome-RealWorld-resnet50-noft.mat'
    elif dset == 'Office10':
        dset_dir = 'Office-Caltech'
        if domain == 'A':
            file_name = 'amazon_decaf.mat'
        elif domain == 'C':
            file_name = 'caltech_decaf.mat'
        elif domain == 'D':
            file_name = 'dslr_decaf.mat'
        elif domain == 'W':
            file_name = 'webcam_decaf.mat'
    elif dset == 'Office31':
        dset_dir = 'Office-31'
        if domain == 'amazon':
            file_name = 'office-A-resnet50-noft.mat'
        elif domain == 'webcam':
            file_name = 'office-W-resnet50-noft.mat'
        elif domain == 'dslr':
            file_name = 'office-D-resnet50-noft.mat'
    elif dset == 'Morden31':
        dset_dir = 'Modern-Office-31'
        if domain == 'amazon':
            file_name = 'Modern-Office-31-amazon-resnet50-noft.mat'
        elif domain == 'webcam':
            file_name = 'Modern-Office-31-webcam-resnet50-noft.mat'
        elif domain == 'synthetic':
            file_name = 'Modern-Office-31-synthetic-resnet50-noft.mat'
    elif dset == 'Office31TIDOT':
            dset_dir = 'Office31_TIDOT_IJCAI21'
            if domain == 'amazon':
                file_name = 'office-A-resnet50-noft-tidot.mat'
            elif domain == 'webcam':
                file_name = 'office-W-resnet50-noft-tidot.mat'
            elif domain == 'dslr':
                file_name = 'office-D-resnet50-noft-tidot.mat'
    elif dset == 'Adaptiope':
        dset_dir = 'Adaptiope'
        if domain == 'Product':
            file_name = 'Adaptiope-product_images-resnet50-noft.mat'
        elif domain == 'Real_Life':
            file_name = 'Adaptiope-real_life-resnet50-noft.mat'
        elif domain == 'Synthetic':
            file_name = 'Adaptiope-synthetic-resnet50-noft.mat'
    elif dset == 'VisDA':  
        dset_dir = 'VisDA-2017'
        if domain == 'synthetic':
            file_name = 'VisDA2017-S-resnet50-noft.mat'
        elif domain == 'realistic':
            file_name = 'VisDA2017-R-resnet50-noft.mat' 
            
    path = os.path.join(data_dir,dset_dir,file_name)
    return path
    

class ImageSet_dataset(Dataset):
    def __init__(self, mat_path, Office10 = False,idx=None):    
        if mat_path == './DATA/VisDA-2017/VisDA2017-S-resnet50-noft.mat':
            data = h5py.File(mat_path)
            h5py_data = h5py.File(mat_path) # the matrix in h5py is the transposed 
            data = {}
            data['resnet50_features'] = np.transpose(h5py_data['resnet50_features'])  # recover the original shape of data matrix
            data['labels'] = np.transpose(h5py_data['labels'])
            del h5py_data
        else:
            data = io.loadmat(mat_path)
        if Office10 is True:
            img = torch.from_numpy(data['feas']).float()
            label = torch.from_numpy((data['labels']-1).squeeze(1)).long()
        else:
            img = torch.from_numpy(data['resnet50_features'])
            img = img.view([img.shape[0],-1])
            label = torch.from_numpy(data['labels'].squeeze(0))
        
        if idx !=None:
            sam_idx = (torch.zeros(len(label))==1)
            for cls_idx in idx:
                sam_idx += (label==cls_idx)
            img = img[sam_idx,:]
            label = label[sam_idx]
        if dset=='VisDA':
            img=img.to(torch.float32)
        
        self.img = img
        self.label = label
        del label, img
        
    def __getitem__(self, idx):
        Batch_data = self.img[idx,:]
        Batch_label = self.label[idx]
        return Batch_data, Batch_label
    
    def __len__(self):
        return len(self.label)
    
def Make_Loader(mat_path, Batch_size = None, Office10 = False, partial = False,shuffle=False, drop_last =False):
    if partial == False:
        data_set = ImageSet_dataset(mat_path, Office10)
    else:
        data_set = ImageSet_dataset(mat_path, Office10,partial_cls_idx)
    if Batch_size is None:
        Batch_size = len(data_set)
    new_loader = DataLoader(data_set, Batch_size, shuffle = shuffle, drop_last = drop_last)
    return new_loader

##################################
# Define Networks
##################################
class FC(nn.Module):
    def __init__(self, input_dim,conv_dim_1, conv_dim_2,):#conv_dim_3):
        super(FC, self).__init__()
        self.fc1 = nn.Sequential( 
              nn.Linear(input_dim, conv_dim_1), 
              nn.BatchNorm1d(conv_dim_1),
              nn.LeakyReLU(negative_slope=0.2, inplace=True), 
             ) 
        self.fc2 = nn.Sequential( 
              nn.Linear(conv_dim_1, conv_dim_2), 
              nn.BatchNorm1d(conv_dim_2),
              nn.Tanh(),
             ) 
 

    def forward(self, x):
        z_1 = self.fc1(x)
        z_2 = self.fc2(z_1)
        #z_3 = self.fc3(z_2)
        return z_1, z_2

    
class C(nn.Module):
    def __init__(self, conv_dim_2,n_classes):
        super(C, self).__init__()
        self.fc3 = nn.Linear(conv_dim_2, n_classes) 

    def forward(self, x):
        y = self.fc3(x)
        p = F.softmax(y)
        return p

     
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0)         
########################         
def reset_grad():
    """Zeros the gradient buffers."""
    c.zero_grad()
    fc.zero_grad()
    
def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()   
    

def classification_accuracy_train():
    with torch.no_grad():
        correct = 0

        iter_loader_s = iter(source_loader)
        X_s, lab_s = iter_loader_s.__next__()
        # if dset=='VisDA':
        #     X_s=X_s.to(torch.float32)
        X_s, lab_s = to_var(X_s), to_var(lab_s).long().squeeze()
        h_s1 , h_s2 = fc(X_s)
        output = c(h_s2)           
        pred = output.data.max(1)[1]
        correct += pred.eq(lab_s.data).cpu().float()
        
        return correct.mean().item()    


def classification_accuracy():
    with torch.no_grad():
        correct = 0
        
        if dset == 'VisDA':
            iter_loader_t = iter(val_loader)
        else:
            iter_loader_t = iter(target_loader)
        X_t, lab_t = iter_loader_t.__next__()
        # if dset=='VisDA':
        #     X_t=X_t.to(torch.float32)
        X_t, lab_t = to_var(X_t), to_var(lab_t).long().squeeze()
        h_t1 , h_t2 = fc(X_t)
        output = c(h_t2)           
        pred = output.data.max(1)[1]

        correct += pred.eq(lab_t.data).cpu().float()
        return correct.mean().item() 
    

def dist(X, Y):
    xx = X.pow(2).sum(1).repeat(Y.shape[0],1)
    xy = X@Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0],1)   
    dist = xx.t() + yy - 2*xy
    
    return dist

def create_im_weights_update(lab_s,pred_s,pred_t, class_num):
    
    lab_s1 = lab_s.cpu().detach().numpy()
    pred_s1 = pred_s.cpu().detach().numpy()
    pred_t1 = pred_t.cpu().detach().numpy()
    cov = np.zeros((class_num,class_num))
    target_y = np.zeros((class_num,1))
    source_y = np.zeros((class_num,1))
    
    for i in range(class_num):
        idx=lab_s1==i
        cov[:,i]=np.sum(pred_s1[idx,:],axis=0)
        target_y[i,0]=np.sum(pred_t1[:,i])
        source_y[i,0]=len(lab_s1[idx])/len(lab_s1)
        
    cov=cov/np.sum(cov)
    target_y=target_y/np.sum(target_y)
    
    im_weights=im_weights_update(source_y, target_y, cov)

    return im_weights
    

def im_weights_update(source_y, target_y, cov):
    dim = cov.shape[0]
    source_y = source_y.astype(np.double)
    target_y = target_y.astype(np.double)
    cov = cov.astype(np.double)

    P = matrix(np.dot(cov.T, cov), tc="d")
    q = -matrix(np.dot(cov.T, target_y), tc="d")
    G = matrix(-np.eye(dim), tc="d")
    h = matrix(np.zeros(dim), tc="d")
    A = matrix(source_y.reshape(1, -1), tc="d")
    b = matrix([1.0], tc="d")
    sol = solvers.qp(P, q, G, h, A, b)
    new_im_weights = np.array(sol["x"])

    return new_im_weights

##########################################################################
# sample from a mixture distribution
##########################################################################


def mix_dis(X_s,num,alpha=1.0):
    
    idx1=torch.multinomial(torch.ones(X_s.shape[0]),num,replacement=True)
    idx2=torch.multinomial(torch.ones(X_s.shape[0]),num,replacement=True)
    
    # "lam = np.random.beta(alpha,alpha)" can be replaced with "lam = alpha", and in this case, "alpha" plays the same role as "theta".
    lam = np.random.beta(alpha,alpha)   
    mixup_X = lam * X_s[idx1,:] + (1-lam) * X_s[idx2,:]
    
    return mixup_X
    
    
def new_source_data(X_s,lab_s,n_sam,alpha=0.2,n_class=n_classes):
   
   
    for i in range(n_class):
        if n_sam[i]!=0:
            idx=lab_s==i
            if i == 0:
               
                X_tmp = X_s[idx,:]
                sam = mix_dis(X_tmp,n_sam[i],alpha)
                lab_sam = (torch.ones(sam.shape[0]).cuda() )*i  
            else:
              
                X_tmp = X_s[idx,:]
                sam_tmp = mix_dis(X_tmp,n_sam[i],alpha)
                lab_sam_tmp = (torch.ones(sam_tmp.shape[0]).cuda() )*i
                sam =torch.cat((sam,sam_tmp),axis=0)
                lab_sam =torch.cat((lab_sam,lab_sam_tmp))                   
        if i == n_class-1:
            return sam,lab_sam

            

##########################################################################
# ETIC (eot-based independence criterion)
##########################################################################
def sinkhorn_divergence(M1, M2, reg, a=None, b=None,numItermax=100, stopThr=1e-8,):
   
    n, m = M1.shape[0], M2.shape[0]
    
    K1 = torch.exp(M1 / (-reg)).detach()
    K2 = torch.exp(M2 / (-reg)).detach()
   
    u ,v = torch.ones(n, m).cuda(),torch.ones(n, m).cuda()
    
    cpt = 0
    err = 1
    
    while (err > stopThr and cpt < numItermax ):    
        u =  a / ( K1 @ v @ (K2.T) )  
        v =  b / ( (K1.T) @ u @ K2 )
   
        cpt = cpt + 1
        err = torch.linalg.norm( u * ( K1 @ v @ (K2.T) )-a)

    cost = torch.sum(u*( (K1*M1) @ v @ K2.T )  )
    
    return cost


def ETIC(X , Y, reg,a):
    
        xmat=dist(X,X)        
        ymat=dist(Y,Y)
        xmat[xmat < 0] = 0
        ymat[ymat < 0] = 0
         
        num=torch.sum(a,axis=0)
        ns=num[0]
        nt=num[1]
        
        b=torch.zeros(X.shape[0],2).cuda()
        b[:,0] = ns/( (ns+nt) )
        b[:,1] = nt/( (ns+nt) )
        
        
        a = a/(ns+nt)
        b = b/(ns+nt)
    
        ymat=ymat
        if torch.median(xmat)!=0:
            xmat=xmat/(4*torch.median(xmat))
        else:
            xmat=xmat/(2*torch.max(xmat))

        cost = sinkhorn_divergence(xmat, ymat, reg, a, b)
        cost1 = sinkhorn_divergence(xmat, ymat, reg, a, a)
        cost2 = sinkhorn_divergence(xmat, ymat, reg, b, b)
   
        return cost - cost1/2 - cost2/2
    
def HSIC(X, Y):
    
    xmat=dist(X,X)
    xmat[xmat < 0] = 0

    m=X.shape[0]
    
    K = torch.exp( xmat/ (-4*torch.median(xmat)))
    L = Y @ Y.T
    H = ( torch.eye(m) - torch.ones(m,m)/m ).cuda()
    
    return  torch.trace(K @ H @ L @ H) / ( (m-1)**2 )
################################### Loss Func ##################################
def Pred_Entropy_loss(pred_t):
    num_sam = pred_t.shape[0]
    Entropy = -(pred_t.mul(pred_t.log()+1e-4)).sum()
    
    return Entropy/num_sam

def One_Hot(lab, num_cls):
    num_sam = lab.shape[0]
    
    lab_mat = torch.zeros([num_sam,num_cls]).cuda().scatter_(1, lab.unsqueeze(1), 1)
    
    return lab_mat

def Weighting_ERM_loss(pred, lab, num_cls, weight_ratio):
    one_hot_lab = One_Hot(lab, num_cls)
    pred_score = (one_hot_lab.mul(pred)).sum(1)
    Cross_Entropy = -(pred_score).log()
    pred_weight = weight_ratio[lab]
    
    weighting_ERM = (Cross_Entropy.mul(pred_weight)).mean()
    
    return weighting_ERM

    
def Feature_Domain_Independence(X,Y,lab_x=None,lab_y=None, w=None,cri = 'ETIC'):
    n1=X.size()[0]
    n2=Y.size()[0]

    Domain_labels=torch.eye(2,2)
    Domain_labels=Domain_labels.cuda()
    Z=torch.cat((X,Y),0)
    
    a=torch.zeros([n1+n2,2]).cuda()
    a[0:n1,0]=1
    a[n1:n1+n2,1]=1
    
    
    FDI = 0
    if cri == 'ETIC':
              
        L=torch.cat((lab_x,lab_y),0)
        for i in range(n_classes):
            Ltmp=L==i
            if torch.sum(Ltmp[n1:n1+n2]!=0)>1:
                Ztmp=Z[Ltmp,:]        
                atmp=a[Ltmp,:]
                Dtmp=Domain_labels
                if w==None:
                    FDI += 1* ETIC(Ztmp,Dtmp,eps,atmp)
                else:
                    FDI += w[i]*ETIC(Ztmp,Dtmp,eps,atmp)
       
            
    elif cri == 'HSIC':
        
        Domain_labels=torch.zeros([n1+n2,2])
        Domain_labels[0:n1,0]=1
        Domain_labels[n1:n1+n2,1]=1
        Domain_labels=Domain_labels.cuda()
        L=torch.cat((lab_x,lab_y),0)
        for i in range(n_classes):
            Ltmp=L==i
            if torch.sum(Ltmp[n1:n1+n2]!=0)>1:
                Ztmp=Z[Ltmp,:]
                Dtmp=Domain_labels[Ltmp,:]
                FDI += HSIC(Ztmp,Dtmp)
                if w==None:
                    FDI += 1* HSIC(Ztmp,Dtmp)
                else:
                    FDI += w[i]*HSIC(Ztmp,Dtmp)
    if w==None:   
        return FDI/n_classes
    else:
        return FDI



########################
Max_ACC = torch.zeros((len(FD_lam_list),len(FD_step),len(pre),Exptimes,len(save_name)))
final_result = np.zeros((Exptimes+7,len(save_name)))


for Domain_iter in range(len(save_name)):
    
    source_domain = source_domain_set[Domain_iter]
    target_domain = target_domain_set[Domain_iter]
    source_path = get_mat_path(dset,source_domain)
    target_path = get_mat_path(dset,target_domain)
    if dset =='VisDA':
        source_loader = Make_Loader(source_path, Batch_size = batch_size, shuffle = True, drop_last = True)
        target_loader = Make_Loader(target_path, Batch_size = batch_size,partial=True, shuffle = True, drop_last = True)
        source_loader1 =Make_Loader(source_path, Batch_size = None,partial=False, shuffle = False,  drop_last = False)
        val_loader = Make_Loader(target_path, Batch_size = None,partial=True, shuffle = False,  drop_last = False)
    else:    
        source_loader = Make_Loader(source_path,Batch_size = None)
        target_loader = Make_Loader(target_path,Batch_size = None,partial=True)
     
    best_origin_acc = 0
    for k in range(len(FD_lam_list)):
     for l in range(len(FD_step)):
      for m in range(len(pre)):
        Exp_iter = 0
        while Exp_iter < Exptimes:
            # Define the network architech   
            torch.manual_seed(Exp_iter)
            if dset=='VisDA' or dset=='OfficeHome':
                fc = FC(2048 , 1042,256) 
                c = C(256,n_classes )
            else: 
                fc = FC(2048 , 1042,512) 
                c = C(512,n_classes )
            fc.cuda()
            c.cuda()
            fc.apply(weights_init)
            c.apply(weights_init)
    
           
            c_solver = optim.Adam([{'params':c.parameters()},{'params':fc.parameters()}], 
                                  lr, [beta1, beta2], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(c_solver, step_size=150, gamma=0.5)
            criterionQ_dis = nn.NLLLoss().cuda() 
                                
            ####################
            # Train procedure
            ####################
            result = np.zeros((6,epochs*max(len(source_loader),len(target_loader))))
            print("************ %1s→%1s: Start Experiment %1s training ************"%(source_domain,target_domain,Exp_iter+1))  
            
            if dset == 'VisDA':
                iter_loader_tmp = iter(source_loader)
                X_s_tmp, lab_s_tmp = iter_loader_tmp.__next__()
                X_s_tmp, lab_s_tmp = Variable(X_s_tmp), Variable(lab_s_tmp)
                w_s = torch.zeros(n_classes,1)
                for i in range(n_classes):
                    idx=lab_s_tmp==i
                    w_s[i,0]= len(lab_s_tmp[idx]) /len(lab_s_tmp)
                w_s = w_s.cuda()
            w2=torch.ones(n_classes,1).cuda()
            
            for step in range(epochs*max(len(source_loader),len(target_loader))):
    
                Current_loss = np.array([0])
                
                if dset == 'VisDA':
                    if step % len(source_loader) == 0:
                        iter_loader_s = iter(source_loader)
                    if step % len(target_loader) == 0:
                        iter_loader_t = iter(target_loader)
                else:
                    iter_loader_s = iter(source_loader)
                    iter_loader_t = iter(target_loader)
                X_s, lab_s = iter_loader_s.__next__()
                X_t, lab_t = iter_loader_t.__next__()
                #fc.train()
                #c.train()
                
                
                lab_s = lab_s.type(torch.LongTensor)
                lab_t = lab_t.type(torch.LongTensor)
                
    
                # if dset =='VisDA':
                #     X_s=X_s.to(torch.float32)
                #     X_t=X_t.to(torch.float32)
            
                X_s, lab_s = Variable(X_s), Variable(lab_s)
                X_s, lab_s = X_s.cuda(), lab_s.cuda()
                X_t, lab_t = Variable(X_t), Variable(lab_t)
                X_t, lab_t = X_t.cuda(), lab_t.cuda()
                
                    
                    # Init gradients
                reset_grad()
                
                
                h_s, h_s2 = fc(X_s)
                h_t, h_t2 = fc(X_t)
                
                
                if dset!='VisDA':
                    w_s = torch.zeros(n_classes,1).cuda()
                    for i in range(n_classes):
                        idx=lab_s==i
                        w_s[i,0]= len(lab_s[idx]) /len(lab_s)
                    
                w_t = (w_s*w2).squeeze(1).detach()
                
                pred_s = c(h_s2)
                pred_t = c(h_t2)
                pred_t_NG = pred_t.detach()
                plab_t = pred_t_NG.data.max(1)[1]     

                
                if step > (FD_step[l]-1):
         
                      #generate new data
                      S = torch.multinomial(w_t,int(2*X_s.shape[0]),replacement=True)
                      n_sam = torch.zeros(n_classes).cuda()
                      for i in range(n_classes):
                          n_sam[i]=torch.sum(S == i)
                     
                      new_X_s,new_lab_s = new_source_data(X_s,lab_s,n_sam.int(),alpha=alpha,n_class=n_classes)
                     
                      new_h_s, new_h_s2 = fc(new_X_s)
                      new_pred_s = c(new_h_s2)
                      new_lab_s = ( new_lab_s.type(torch.LongTensor) ).cuda()
                      new_Ls =torch.zeros(new_lab_s.shape[0], n_classes).cuda().scatter_(1, torch.unsqueeze(new_lab_s, 1), 1)
                           
                #==========================================================
                #                     Loss Part
                #==========================================================
   
                
                if step <= (FD_step[l] - 1):
                    
                    FD_loss = torch.zeros(1).squeeze(0).cuda()           
                    CE_loss = criterionQ_dis(torch.log(pred_s), lab_s)
                else:                 
                    if step >= (FD_step[l]+pre[m]):               
                        CE_loss = criterionQ_dis(torch.log(new_pred_s), new_lab_s)
                        #CE_loss = Weighting_ERM_loss(pred_s, lab_s, n_classes, w2)
                        FD_loss = \
                        FD_lam_list[k]*Feature_Domain_Independence(h_s2,h_t2,lab_s,plab_t,w=w_t,cri = 'ETIC')                    
                    else:
                        CE_loss = criterionQ_dis(torch.log(pred_s), lab_s)                  
                        FD_loss = \
                        FD_lam_list[k]*Feature_Domain_Independence(h_s2,h_t2,lab_s,plab_t,w=None,cri = 'ETIC')

                #================ Final Objective ==============
                c_loss =  CE_loss  + FD_loss 
                #================ Final Objective  ==============
                
                
                c_loss.backward()
                c_solver.step()
                #scheduler.step()
               
                c.eval()
                fc.eval()
    
                if step > (FD_step[l]) and (step < (FD_step[l]+pre[m])):
                    w2 = 0.5*w2 + 0.5*torch.tensor(create_im_weights_update(lab_s,pred_s.detach(),pred_t.detach(), n_classes)).cuda()#0.5*w2 + 0.5*
                elif   step >= (FD_step[l]+pre[m]): 
                    w2 = 0.5*w2 + 0.5*torch.tensor(create_im_weights_update(lab_s,pred_s.detach(),pred_t.detach(), n_classes)).cuda()#0.5*w2 + 0.5*
                
        
                print('========================== Testing start! ===========================')
                
                
                if dset == 'VisDA' and step %10 == 0:
                    val_acc = classification_accuracy()
                    train_acc = classification_accuracy_train()
                    
                elif dset != 'VisDA':    
                    val_acc = classification_accuracy()   
                    train_acc = classification_accuracy_train()
                
                
                if val_acc > best_origin_acc:
                    best_origin_acc = val_acc                
                
                
                Current_loss = c_loss.cpu().detach().numpy()    
                result[:,step] = [1,1,1,Current_loss,train_acc,val_acc]
 
                print('====================== %1s→%1s: Experiment %1s Epoch %1s ================='%(source_domain,target_domain,Exp_iter+1,step+1))
                print('Validation accuracy: {}'.format(val_acc))     
                print('Current Max Val_Accuracy: {}'.format(max(result[5,:])))
                print('Train accuracy: {}'.format(train_acc))
                print('Loss_risk: {}'.format(CE_loss))
                print('Loss_align: {}'.format(FD_loss))
                print('Total_Loss: {}'.format(Current_loss))
         
                print('mu: {}'.format(FD_lam_list[k]))
                print('FD_step: {}'.format(FD_step[l]))    
                print('pre: {}'.format(pre[m]))    
                print('alpha: {}'.format(alpha))             
                          
                if max(result[5,:]) == 1:
                    print('Reach accuracy {1} at Epoch %1s !'%(step+1))
                    break
                
            
           
            #print('==============================Experiment Finish==================')
            Max_ACC[k,l,m,Exp_iter,Domain_iter] = max(result[5,:])     

            print("************ %1s→%1s: End Experiment %1s training ************"%(source_domain,target_domain,Exp_iter+1))
            
            Exp_iter += 1
    
                
tmp_ACC,_= torch.max(Max_ACC,axis=3)
for i in range(len(save_name)):
    tmp = tmp_ACC[:,:,:,i]
    max_index=torch.nonzero(tmp==torch.max(tmp))
    final_result[Exptimes,i] = FD_lam_list[max_index[0,0].item()]
    final_result[Exptimes+1,i] = FD_step[max_index[0,1].item()]
    final_result[Exptimes+2,i] = pre[max_index[0,2].item()]
    final_result[Exptimes+3,:] = eps
    final_result[Exptimes+4,:] = lr
    final_result[Exptimes+5,:] = alpha
    final_result[Exptimes+6,:] = epochs
    final_result[0:Exptimes,i] = Max_ACC[max_index[0,0].item(),max_index[0,1].item(),max_index[0,2].item(),:,i]

io.savemat('%1s_Results.mat'%(dset), {'ACC': final_result}) 
print("Task:",save_name)
print("The final_result: %1s  "%(np.max(final_result[0:Exptimes,:], axis=0)))
#io.savemat('%1s_Results.mat'%(dset), {'ACC': Max_ACC}) 


