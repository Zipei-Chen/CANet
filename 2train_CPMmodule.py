# from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import PIL
import math
import copy
from CPM import CPMmodule
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import functional as TF

# Training settings
parser = argparse.ArgumentParser(description='CPM module')
# Model options

parser.add_argument('--suffix', type=str,
                    default='test',
                    help='name of experiments')
parser.add_argument('--dataroot', type=str,
                    default='./data/',
                    help='path to dataset')
parser.add_argument('--enable_logging', type=bool, default=True)
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--num-workers', default=1, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
# parser.add_argument('--act-decay', type=float, default=0,
#                     help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 10.0)')
parser.add_argument('--fliprot', type=bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=2021, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class TotalDatasetsLoader(data.Dataset):

    def __init__(self, datasets_path, train = True, transform = None, fliprot = False, *arg, **kw):
        super(TotalDatasetsLoader, self).__init__()

        self.dataset = list(torch.load(os.path.join(datasets_path, "generated_dataset_shadow2non_match.pt")))
        self.dataset += list(torch.load(os.path.join(datasets_path, "generated_dataset_non2shaodw_match.pt")))
        self.dataset += list(torch.load(os.path.join(datasets_path, "generated_dataset_shadow2shadow_match.pt")))

        non_match_dataset = list(torch.load(os.path.join(datasets_path, "generated_dataset_shadow2non_non_match.pt")))
        non_match_dataset += list(torch.load(os.path.join(datasets_path, "generated_dataset_non2shaodw_non_match.pt")))
        non_match_dataset += list(torch.load(os.path.join(datasets_path, "generated_dataset_shadow2shadow_non_match.pt")))

        self.dataset += random.sample(non_match_dataset, len(self.dataset))

        del non_match_dataset

        self.train = train
        self.transform = transform
        self.fliprot = fliprot

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img

        if self.train:
            instance = self.dataset[index]
            c = instance[2]
            y, x = divmod(int(instance[1]), 13)
            y2, x2 = divmod(int(instance[2]), 13)
            image_a = PIL.Image.open(instance[0]).convert('RGB').resize([416, 416])
            image_b = PIL.Image.open(os.path.join("F:/ISTD_Dataset/train_illumination_independent", os.path.basename(instance[0]))).convert('RGB').resize([416, 416])

            patch_a1 = image_a.crop((x, y, x + 32, y + 32))
            patch_a2 = image_b.crop((x, y, x + 32, y + 32))

            patch_b1 = image_a.crop((x2, y2, x2 + 32, y2 + 32))
            patch_b2 = image_b.crop((x2, y2, x2 + 32, y2 + 32))

            patch_a1 = transform_img(patch_a1)
            patch_a2 = transform_img(patch_a2)
            patch_b1 = transform_img(patch_b1)
            patch_b2 = transform_img(patch_b2)

            patch_a = torch.cat([patch_a1, patch_a2], dim=0)
            patch_b = torch.cat([patch_b1, patch_b2], dim=0)

            # transform images if required
            if self.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5

                if do_rot:
                    patch_a = patch_a.permute(0,2,1)
                    patch_b = patch_b.permute(0,2,1)

                if do_flip:
                    patch_a = torch.from_numpy(deepcopy(patch_a.numpy()[:,:,::-1]))
                    patch_b = torch.from_numpy(deepcopy(patch_b.numpy()[:,:,::-1]))

            label_type = torch.tensor(float(instance[3]))
            label_correlation = torch.tensor(float(instance[4]))

            return patch_a, patch_b, label_type, label_correlation

    def __len__(self):
            if self.train:
                return len(self.dataset)


def create_loaders():
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_loader = torch.utils.data.DataLoader(
        TotalDatasetsLoader(train=True,
                         datasets_path=args.dataroot,
                         fliprot=args.fliprot,
                         transform=transform_train),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return train_loader


def train(train_loader, model, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.L1Loss()
    for batch_idx, data in pbar:
        optimizer.zero_grad()
        patch_a, patch_b, label_type, label_correlation = data

        if args.cuda:
            patch_a, patch_b = patch_a.cuda(), patch_b.cuda()
            label_type, label_correlation = label_type.cuda(), label_correlation.cuda()
            patch_a, patch_b = Variable(patch_a), Variable(patch_b)
            patch_type, correlation_degree = model(patch_a, patch_b)

        classify_loss = criterion1(patch_type, label_type.long())
        regress_loss = criterion2(correlation_degree, label_correlation)

        loss_total = classify_loss + regress_loss
        loss_total.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, len(train_loader))

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(patch_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss_total.item()))
    if (args.enable_logging):
        logger.log_value('loss', loss_total.item()).step()
    try:
        os.stat('{}{}'.format(args.model_dir, args.suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir, args.suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, args.suffix, epoch))


def adjust_learning_rate(optimizer, length):
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
            1.0 - float(group['step']) * float(args.batch_size) / (float(args.epochs) * length * args.batch_size))
    return


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch, logger)

if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, args.suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    logger, file_logger = None, None
    model = CPMmodule()
    if (args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)
    train_loader = create_loaders()
    main(train_loader, model, logger, file_logger)
