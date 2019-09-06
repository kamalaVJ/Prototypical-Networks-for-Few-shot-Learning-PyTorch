# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from chinadrinks_dataset import ChinadrinkDataset
from protonet import ProtoNet
from parser_util_extract import get_parser
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import shutil
import pickle

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode, root):
    dataset = ChinadrinkDataset(mode=mode, root= root, size = opt.img_size)
    n_classes = len(np.unique(dataset.y))
    print(n_classes)
    #print(dataset.y)
    if  n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        #classes_per_it = 1034
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        #classes_per_it = 434
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode, root):
    dataset = init_dataset(opt, mode, root)
    #labels = [int(x) for x in dataset.y]
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    torch.cuda.empty_cache()
    return dataloader


def init_protonet(opt, pretrained_file= "", pretrained = False):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    if(pretrained):
        model.load_state_dict(torch.load(pretrained_file))
        print("Loaded pre-trained model")
    return model



def train(opt, tr_dataloader, model):
    '''
    Train the model with the prototypical learning algorithm
    '''
    #writer = SummaryWriter('/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/logs/Chinadrink_Protonet_22_dropout')
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        torch.cuda.empty_cache()
        for batch in tqdm(tr_iter):

            #optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            


    return model_output,y


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    #writer = SummaryWriter('/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/logs/Chinadrink_Protonet_22_dropout')
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
                
    

    return model_output,y


def eval(opt):
    '''
    Initialize everything and train
    '''
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    #dataset_root = options.dataset_root
    

    train_folder ='/home/caffe/data/chinadrink_prod_train'
    test_folder = '/home/caffe/data/chinadrink_test/all_cropped_images/'

    
    filepath = '/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/output/best_model_rgb28.pth'
    train_features_file = '/home/caffe/data/chinadrink'
    test_features_file = '/home/caffe/data/chinadrink'

    
    '''
    tr_dataloader = init_dataloader(options, 'train', root = train_folder)
    '''
    
    model = init_protonet(opt = options,pretrained_file = filepath,  pretrained = True)
    '''
    train_features, train_labels = train(opt=options,
                tr_dataloader=tr_dataloader,
                model=model)
    
    #save train features
    np.save(train_features_file+'train_features_rgb', train_features.cpu().detach().numpy())
    #save train labels
    with open(train_features_file+'train_labels_rgb.pkl','wb') as f:
        pickle.dump(train_labels.cpu(),f)
    print('Loaded train features/labels')
    '''
    test_dataloader = init_dataloader(options, 'test', root = test_folder)
    
    test_features, test_labels = test(opt=options,
         test_dataloader=test_dataloader,
         model=model)
    print(test_features.size())
    #save test features
    np.save(test_features_file+'test_features_rgb', test_features.cpu().detach().numpy())
    #save test labels
    with open(test_features_file+'test_labels_rgb.pkl','wb') as f:
        pickle.dump(test_labels.cpu(),f)

    
    

   


if __name__ == '__main__':
    main()
