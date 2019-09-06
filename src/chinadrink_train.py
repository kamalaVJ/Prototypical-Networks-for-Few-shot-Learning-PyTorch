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
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
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


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    writer = SummaryWriter('/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/logs/Chinadrink_Protonet_rgb64')
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    torch.cuda.empty_cache()
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        torch.cuda.empty_cache()
        for batch in tqdm(tr_iter):

            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            writer.add_scalar('Training loss per episode',loss.item(),len(train_loss))
            writer.add_scalar('Training accuracy per episode', acc.item(), len(train_acc))

        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])

        writer.add_scalar('Training loss per epoch',avg_loss, epoch+1)
        writer.add_scalar('Training accuracy per epoch',avg_acc, epoch+1)

        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        
        writer.add_scalar('Learning rate', lr_scheduler.get_lr()[0],len(train_loss)) 

        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)

            torch.cuda.empty_cache()
            val_loss.append(loss.item())
            val_acc.append(acc.item())

            writer.add_scalar('Validation loss per episode',loss.item(),len(val_loss))
            writer.add_scalar('Validation accuracy per episode',acc.item(),len(val_acc))

        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])

        writer.add_scalar('Validation loss per epoch',avg_loss,epoch+1)
        writer.add_scalar('Validation accuracy per epoch',avg_acc,epoch+1)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
    
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    writer = SummaryWriter('/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/logs/Chinadrink_Protonet_rgb64')
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
    writer.add_scalar("Test accuracy",avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


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

def split_dataset(filepath, files):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

    #print(filepath) 
    for img in files:
        
        path = os.path.join(filepath,os.path.basename(os.path.dirname(img)))
        #print(path) 
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(img,path)


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

    dataset_root = options.dataset_root
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2

    image_folders = [os.path.join(dataset_root, SKU, image) for SKU in os.listdir(dataset_root) \
                    if os.path.isdir(os.path.join(dataset_root, SKU)) for image in os.listdir(os.path.join(dataset_root, SKU))]

    total_num = len(image_folders)
    train_num = int(train_split*total_num)
    val_num = int(val_split*total_num)
    test_num = total_num - (train_num + val_num)

    random.seed(0)
    random.shuffle(image_folders)

    metatrain = image_folders[:train_num]
    print("No: of Images in train:", len(metatrain))
    metaval = image_folders[train_num:train_num+val_num]
    print("No: of images in validation: ", len(metaval))
    metatest = image_folders[train_num+val_num:]
    print("No: of images in test: ", len(metatest))

    root_folder = '/home/caffe/data/'

    train_folder = root_folder +'train'
    val_folder = root_folder+'val'
    test_folder = root_folder+'test'

    split_dataset(train_folder, metatrain)
    split_dataset(val_folder, metaval)
    split_dataset(test_folder, metatest)
    


    tr_dataloader = init_dataloader(options, 'train', root = train_folder)
    torch.cuda.empty_cache()
    
    val_dataloader = init_dataloader(options, 'val', root = val_folder)

    torch.cuda.empty_cache()
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test', root = test_folder)

    torch.cuda.empty_cache()

    filepath = '/home/caffe/orbix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/output/best_model_rgb28.pth'

    model = init_protonet(opt = options,pretrained_file = filepath,  pretrained = True)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()
