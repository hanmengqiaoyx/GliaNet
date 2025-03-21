import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from utils import progress_bar
from cutout import *
from contra_loss import *
from model import ResNet18
from Astrocyte_Network import Pre_head, Oli_Network_1, Oli_Network_2, Oli_Network_3, Oli_Network_4, \
    Astro_Network_1, Astro_Network_2, Astro_Network_3, Astro_Network_4


parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr_Net', default=1e-1, type=float, help='learning rate')
parser.add_argument('--lr_Glial', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
parser.add_argument('--epoch', default=1000, type=int, help='max epoch')
args = parser.parse_args()
gpu = '0,3,4'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1
train_loss_Glial = 1000
train_loss_Net = 1000


# Data Loading
def data_prepare():
    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../../../../../../../data/hanmq/CIFAR10', train=True, download=False, transform=TwoCropTransform(transform1, transform2))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='../../../../../../../../data/hanmq/CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# Model Loading
def model_prepare():
    Net = ResNet18()
    Net.to(device)
    Head = Pre_head()
    Head.to(device)
    ON_1 = Oli_Network_1()
    ON_1.to(device)
    ON_2 = Oli_Network_2()
    ON_2.to(device)
    ON_3 = Oli_Network_3()
    ON_3.to(device)
    ON_4 = Oli_Network_4()
    ON_4.to(device)
    AN_1 = Astro_Network_1()
    AN_1.to(device)
    AN_2 = Astro_Network_2()
    AN_2.to(device)
    AN_3 = Astro_Network_3()
    AN_3.to(device)
    AN_4 = Astro_Network_4()
    AN_4.to(device)
    Net = torch.nn.DataParallel(Net)
    Head = torch.nn.DataParallel(Head)
    ON_1 = torch.nn.DataParallel(ON_1)
    ON_2 = torch.nn.DataParallel(ON_2)
    ON_3 = torch.nn.DataParallel(ON_3)
    ON_4 = torch.nn.DataParallel(ON_4)
    AN_1 = torch.nn.DataParallel(AN_1)
    AN_2 = torch.nn.DataParallel(AN_2)
    AN_3 = torch.nn.DataParallel(AN_3)
    AN_4 = torch.nn.DataParallel(AN_4)

    optimizer_Net = optim.SGD(Net.parameters(), lr=args.lr_Net, weight_decay=5e-4, momentum=0.9)
    scheduler_Net = optim.lr_scheduler.ReduceLROnPlateau(optimizer_Net, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_Net = optim.lr_scheduler.MultiStepLR(optimizer_Net, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_head = optim.SGD(Head.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_head = optim.lr_scheduler.ReduceLROnPlateau(optimizer_head, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_head = optim.lr_scheduler.MultiStepLR(optimizer_head, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_ON_1 = optim.SGD(ON_1.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_ON_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ON_1, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_ON_1 = optim.lr_scheduler.MultiStepLR(optimizer_ON_1, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_ON_2 = optim.SGD(ON_2.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_ON_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ON_2, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_ON_2 = optim.lr_scheduler.MultiStepLR(optimizer_ON_2, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_ON_3 = optim.SGD(ON_3.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_ON_3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ON_3, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_ON_3 = optim.lr_scheduler.MultiStepLR(optimizer_ON_3, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_ON_4 = optim.SGD(ON_4.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_ON_4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ON_4, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_ON_4 = optim.lr_scheduler.MultiStepLR(optimizer_ON_4, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_AN_1 = optim.SGD(AN_1.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_1, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_AN_1 = optim.lr_scheduler.MultiStepLR(optimizer_AN_1, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_AN_2 = optim.SGD(AN_2.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_2, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_AN_2 = optim.lr_scheduler.MultiStepLR(optimizer_AN_2, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_AN_3 = optim.SGD(AN_3.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_3, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_AN_3 = optim.lr_scheduler.MultiStepLR(optimizer_AN_3, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    optimizer_AN_4 = optim.SGD(AN_4.parameters(), lr=args.lr_Glial, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_4, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    # scheduler_AN_4 = optim.lr_scheduler.MultiStepLR(optimizer_AN_4, milestones=[100, 175, 250, 300], gamma=0.1, last_epoch=-1)
    ce_criterion = nn.CrossEntropyLoss()
    contra_criterion = Contra_Loss()

    # Net.load_state_dict(torch.load('./Model/Net.t7')['net'])
    # Head.load_state_dict(torch.load('./Model/Head.t7')['net'])
    # ON_1.load_state_dict(torch.load('./Model/ON_1.t7')['net'])
    # ON_2.load_state_dict(torch.load('./Model/ON_2.t7')['net'])
    # ON_3.load_state_dict(torch.load('./Model/ON_3.t7')['net'])
    # ON_4.load_state_dict(torch.load('./Model/ON_4.t7')['net'])
    # AN_1.load_state_dict(torch.load('./Model/AN_1.t7')['net'])
    # AN_2.load_state_dict(torch.load('./Model/AN_2.t7')['net'])
    # AN_3.load_state_dict(torch.load('./Model/AN_3.t7')['net'])
    # AN_4.load_state_dict(torch.load('./Model/AN_4.t7')['net'])
    return Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4, optimizer_Net, scheduler_Net, optimizer_head, scheduler_head, \
           optimizer_ON_1, scheduler_ON_1, optimizer_ON_2, scheduler_ON_2, optimizer_ON_3, scheduler_ON_3, optimizer_ON_4, scheduler_ON_4, \
           optimizer_AN_1, scheduler_AN_1, optimizer_AN_2, scheduler_AN_2, optimizer_AN_3, scheduler_AN_3, optimizer_AN_4, scheduler_AN_4, ce_criterion, contra_criterion


# Training
def train(epoch, dataloader, Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4, optimizer_Net, optimizer_Head, optimizer_ON_1, optimizer_ON_2,
          optimizer_ON_3, optimizer_ON_4, optimizer_AN_1, optimizer_AN_2, optimizer_AN_3, optimizer_AN_4, ce_criterion, contra_criterion, vali=True):
    print('\nEpoch: %d' % epoch)
    global train_loss_Glial, train_loss_Net
    Net.train()
    Head.train()
    ON_1.train()
    ON_2.train()
    ON_3.train()
    ON_4.train()
    AN_1.train()
    AN_2.train()
    AN_3.train()
    AN_4.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    train_loss0 = 0
    correct0 = 0
    total0 = 0
    for batch_id, (img, labels) in enumerate(dataloader):
        # if batch_id < (64 / args.batch_size):
            batch = labels.size(0)
            inputs = torch.cat([img[0], img[1]], dim=0)
            inputs, labels = inputs.to(device), labels.to(device)
            if epoch < 6:
                num_id += 1
                pattern = 0
                i = 0
                outputs, feat_list = Net(inputs, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_Net.zero_grad()
                loss.backward()
                optimizer_Net.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))
            elif epoch >= 6:
                num_id += 1
                inputs_0 = img[0]
                inputs_0, labels = inputs_0.to(device), labels.to(device)
###### Glial-1 ######
                i = 1
                for params_net in Net.parameters():
                    params_net.requires_grad = False
                for params_head in Head.parameters():
                    params_head.requires_grad = True
                for params_on1 in ON_1.parameters():
                    params_on1.requires_grad = True
                for params_on2 in ON_2.parameters():
                    params_on2.requires_grad = False
                for params_on3 in ON_3.parameters():
                    params_on3.requires_grad = False
                for params_on4 in ON_4.parameters():
                    params_on4.requires_grad = False
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = False
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = False
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = False
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = False
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                loss = ce_criterion(outputs, labels.long())
                optimizer_Head.zero_grad()
                optimizer_ON_1.zero_grad()
                loss.backward()
                optimizer_Head.step()
                optimizer_ON_1.step()
                i = 5
                for params_head in Head.parameters():
                    params_head.requires_grad = False
                for params_on1 in ON_1.parameters():
                    params_on1.requires_grad = False
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs, feat_list = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_1.zero_grad()
                loss.backward()
                optimizer_AN_1.step()
###### Glial-2 ######
                i = 2
                for params_head in Head.parameters():
                    params_head.requires_grad = True
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = False
                for params_on2 in ON_2.parameters():
                    params_on2.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                loss = ce_criterion(outputs, labels.long())
                optimizer_Head.zero_grad()
                optimizer_ON_2.zero_grad()
                loss.backward()
                optimizer_Head.step()
                optimizer_ON_2.step()
                i = 6
                for params_head in Head.parameters():
                    params_head.requires_grad = False
                for params_on2 in ON_2.parameters():
                    params_on2.requires_grad = False
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs, feat_list = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_2.zero_grad()
                loss.backward()
                optimizer_AN_2.step()
###### Glial-3 ######
                i = 3
                for params_head in Head.parameters():
                    params_head.requires_grad = True
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = False
                for params_on3 in ON_3.parameters():
                    params_on3.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                loss = ce_criterion(outputs, labels.long())
                optimizer_Head.zero_grad()
                optimizer_ON_3.zero_grad()
                loss.backward()
                optimizer_Head.step()
                optimizer_ON_3.step()
                i = 7
                for params_head in Head.parameters():
                    params_head.requires_grad = False
                for params_on3 in ON_3.parameters():
                    params_on3.requires_grad = False
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs, feat_list = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_3.zero_grad()
                loss.backward()
                optimizer_AN_3.step()
###### Glial-4 ######
                i = 4
                for params_head in Head.parameters():
                    params_head.requires_grad = True
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = False
                for params_on4 in ON_4.parameters():
                    params_on4.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                loss = ce_criterion(outputs, labels.long())
                optimizer_Head.zero_grad()
                optimizer_ON_4.zero_grad()
                loss.backward()
                optimizer_Head.step()
                optimizer_ON_4.step()
                i = 8
                for params_head in Head.parameters():
                    params_head.requires_grad = False
                for params_on4 in ON_4.parameters():
                    params_on4.requires_grad = False
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs_0, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 2
                outputs, feat_list = Net(inputs_0, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_4.zero_grad()
                loss.backward()
                optimizer_AN_4.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))
                i = 0
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = False
                for params_net in Net.parameters():
                    params_net.requires_grad = True
                pattern = 1
                sp_1, sp_2, sp_3, sp_4 = Net(inputs, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                sel1 = ON_1(sel1)
                sel2 = ON_2(sel2)
                sel3 = ON_3(sel3)
                sel4 = ON_4(sel4)
                pr1 = AN_1(pr1, i)
                pr2 = AN_2(pr2, i)
                pr3 = AN_3(pr3, i)
                pr4 = AN_4(pr4, i)
                pattern = 3
                outputs, feat_list = Net(inputs, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                outputs = outputs[:batch]
                loss = ce_criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_Net.zero_grad()
                loss.backward()
                optimizer_Net.step()
                train_loss0 += loss.item()
                _, predicted = outputs.max(1)
                total0 += labels.size(0)
                correct0 += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss0 / num_id, 100. * correct0 / total0, correct0, total0))
        # else:
        #     print('End of the train')
        #     break
    if vali is True:
        if epoch >= 6:
            train_loss_Glial = train_loss / num_id
            train_loss_Net = train_loss0 / num_id
    if epoch < 6:
        return train_loss / num_id, 100. * correct / total, train_loss / num_id, 100. * correct / total
    if epoch >= 6:
        return train_loss / num_id, 100. * correct / total, train_loss0 / num_id, 100. * correct0 / total0


# Testing
def test(epoch, dataloader, Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4, ce_criterion):
    Net.eval()
    Head.eval()
    ON_1.eval()
    ON_2.eval()
    ON_3.eval()
    ON_4.eval()
    AN_1.eval()
    AN_2.eval()
    AN_3.eval()
    AN_4.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(dataloader):
            # if batch_id < (64 / args.batch_size):
                inputs, labels = inputs.to(device), labels.to(device)
                if epoch < 6:
                    num_id += 1
                    pattern = 0
                    i = 0
                    outputs = Net(inputs, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                    loss = ce_criterion(outputs, labels.long())
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / num_id, 100. * correct / total, correct, total))
                elif epoch >= 6:
                    num_id += 1
                    i = 0
                    pattern = 1
                    sp_1, sp_2, sp_3, sp_4 = Net(inputs, 0, 0, 0, 0, 0, 0, 0, 0, pattern, i)
                    sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4 = Head(sp_1, sp_2, sp_3, sp_4)
                    sel1 = ON_1(sel1)
                    sel2 = ON_2(sel2)
                    sel3 = ON_3(sel3)
                    sel4 = ON_4(sel4)
                    pr1 = AN_1(pr1, i)
                    pr2 = AN_2(pr2, i)
                    pr3 = AN_3(pr3, i)
                    pr4 = AN_4(pr4, i)
                    pattern = 3
                    outputs = Net(inputs, sel1, pr1, sel2, pr2, sel3, pr3, sel4, pr4, pattern, i)
                    loss = ce_criterion(outputs, labels.long())
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / num_id, 100. * correct / total, correct, total))
            # else:
            #     print('End of the test')
            # break
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    print('==> Preparing data..')
    trainloader, testloader = data_prepare()

    print('==> Building model..')
    Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4, optimizer_Net, scheduler_Net, optimizer_head, scheduler_head, optimizer_ON_1, \
    scheduler_ON_1, optimizer_ON_2, scheduler_ON_2, optimizer_ON_3, scheduler_ON_3, optimizer_ON_4, scheduler_ON_4, optimizer_AN_1, scheduler_AN_1, \
    optimizer_AN_2, scheduler_AN_2, optimizer_AN_3, scheduler_AN_3, optimizer_AN_4, scheduler_AN_4, ce_criterion,  contra_criterion = model_prepare()

    print('==> Training..')
    train_Glial_loss_lst, train_Glial_acc_lst, train_Net_loss_lst, train_Net_acc_lst, test_loss_lst, test_acc_lst = [], [], [], [], [], []
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train_Glial_loss, train_Glial_acc, train_Net_loss, train_Net_acc = train(epoch, trainloader, Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4,
                                                                                                        optimizer_Net, optimizer_head, optimizer_ON_1, optimizer_ON_2, optimizer_ON_3,
                                                                                                        optimizer_ON_4, optimizer_AN_1, optimizer_AN_2, optimizer_AN_3, optimizer_AN_4,
                                                                                                        ce_criterion, contra_criterion)
        test_loss, test_acc = test(epoch, testloader, Net, Head, ON_1, ON_2, ON_3, ON_4, AN_1, AN_2, AN_3, AN_4, ce_criterion)
        if epoch < 6:
            pass
        elif epoch >= 6:
            scheduler_head.step(train_loss_Glial)
            scheduler_ON_1.step(train_loss_Glial)
            scheduler_ON_2.step(train_loss_Glial)
            scheduler_ON_3.step(train_loss_Glial)
            scheduler_ON_4.step(train_loss_Glial)
            scheduler_AN_1.step(train_loss_Glial)
            scheduler_AN_2.step(train_loss_Glial)
            scheduler_AN_3.step(train_loss_Glial)
            scheduler_AN_4.step(train_loss_Glial)
            scheduler_Net.step(train_loss_Net)
            lr_Glial = optimizer_AN_4.param_groups[0]['lr']
            lr_Net = optimizer_Net.param_groups[0]['lr']
            # print('Glial:', lr_Glial)
            # print('Net:', lr_Net)

            train_Glial_loss_lst.append(train_Glial_loss)
            train_Glial_acc_lst.append(train_Glial_acc)
            train_Net_loss_lst.append(train_Net_loss)
            train_Net_acc_lst.append(train_Net_acc)
            test_loss_lst.append(test_loss)
            test_acc_lst.append(test_acc)
            print('Saving:')
            plt.figure(num=1, dpi=800)
            plt.subplot(2, 3, 1)
            picture1, = plt.plot(np.arange(0, len(train_Glial_loss_lst)), train_Glial_loss_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture1], labels=['ON_loss'], loc='best')
            plt.subplot(2, 3, 2)
            picture2, = plt.plot(np.arange(0, len(train_Glial_acc_lst)), train_Glial_acc_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture2], labels=['ON_acc'], loc='best')
            plt.subplot(2, 3, 3)
            picture3, = plt.plot(np.arange(0, len(train_Net_loss_lst)), train_Net_loss_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture3], labels=['Net_loss'], loc='best')
            plt.subplot(2, 3, 4)
            picture4, = plt.plot(np.arange(0, len(train_Net_acc_lst)), train_Net_acc_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture4], labels=['Net_acc'], loc='best')
            plt.subplot(2, 3, 5)
            picture3, = plt.plot(np.arange(0, len(test_loss_lst)), test_loss_lst, color='green', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture3], labels=['test_loss'], loc='best')
            plt.subplot(2, 3, 6)
            picture4, = plt.plot(np.arange(0, len(test_acc_lst)), test_acc_lst, color='green', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture4], labels=['test_acc'], loc='best')
            plt.savefig('./GlialNet.png')

            # if epoch == args.epoch - 1:
            if lr_Net > 5e-6 or lr_Glial > 5e-6:
                print('Saving:')
                state1 = {
                    'net': Net.state_dict()
                }
                state2 = {
                    'net': Head.state_dict()
                }
                state3 = {
                    'net': ON_1.state_dict()
                }
                state4 = {
                    'net': ON_2.state_dict()
                }
                state5 = {
                    'net': ON_3.state_dict()
                }
                state6 = {
                    'net': ON_4.state_dict()
                }
                state7 = {
                    'net': AN_1.state_dict()
                }
                state8 = {
                    'net': AN_2.state_dict()
                }
                state9 = {
                    'net': AN_3.state_dict()
                }
                state10 = {
                    'net': AN_4.state_dict()
                }
                if not os.path.isdir('Model'):
                    os.mkdir('Model')
                torch.save(state1, './Model/Net''.t7')
                torch.save(state2, './Model/Head''.t7')
                torch.save(state3, './Model/ON_1''.t7')
                torch.save(state4, './Model/ON_2''.t7')
                torch.save(state5, './Model/ON_3''.t7')
                torch.save(state6, './Model/ON_4''.t7')
                torch.save(state7, './Model/AN_1''.t7')
                torch.save(state8, './Model/AN_2''.t7')
                torch.save(state9, './Model/AN_3''.t7')
                torch.save(state10, './Model/AN_4''.t7')
                acc = open('./GlialNet.txt', 'w')
                acc.write(str(test_acc))
                acc.close()
            else:
                print('Saving:')
                state1 = {
                    'net': Net.state_dict()
                }
                state2 = {
                    'net': Head.state_dict()
                }
                state3 = {
                    'net': ON_1.state_dict()
                }
                state4 = {
                    'net': ON_2.state_dict()
                }
                state5 = {
                    'net': ON_3.state_dict()
                }
                state6 = {
                    'net': ON_4.state_dict()
                }
                state7 = {
                    'net': AN_1.state_dict()
                }
                state8 = {
                    'net': AN_2.state_dict()
                }
                state9 = {
                    'net': AN_3.state_dict()
                }
                state10 = {
                    'net': AN_4.state_dict()
                }
                if not os.path.isdir('Model'):
                    os.mkdir('Model')
                torch.save(state1, './Model/Net''.t7')
                torch.save(state2, './Model/Head''.t7')
                torch.save(state3, './Model/ON_1''.t7')
                torch.save(state4, './Model/ON_2''.t7')
                torch.save(state5, './Model/ON_3''.t7')
                torch.save(state6, './Model/ON_4''.t7')
                torch.save(state7, './Model/AN_1''.t7')
                torch.save(state8, './Model/AN_2''.t7')
                torch.save(state9, './Model/AN_3''.t7')
                torch.save(state10, './Model/AN_4''.t7')
                acc = open('./GlialNet.txt', 'w')
                acc.write(str(test_acc))
                acc.close()
                break