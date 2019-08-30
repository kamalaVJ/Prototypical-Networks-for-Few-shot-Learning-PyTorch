# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from chinadrinks_dataset import ChinadrinkDataset
from protonet import ProtoNet
from parser_util import get_parser
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import shutil

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
    #print(type(n_classes))
    #print(dataset.y)
    if n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
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



def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


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


    test_folder = '/home/caffe/data/chinadrink_test/all_cropped_images/'

    filepath = '/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/output/best_model_65.pth'


    
    torch.cuda.empty_cache()
    
    test_dataloader = init_dataloader(options, 'test', root = test_folder)

    torch.cuda.empty_cache()

    model = init_protonet(opt = options,pretrained_file = filepath,  pretrained = True)
    
    print('Testing with the best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    



if __name__ == '__main__':
    main()
