#!/usr/local/ancoonda3/envs/nas/bin/python
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np

def DataLoader(datapath='/home/while1training/DATA/NLOS/', rooms=['restroom', '526', '511', 'room4'], nodes=[1, 2, 3, 4], train_part=0.8, world_size=None, rank=None, batch_size=64, random_seed=0):
    trainset = []
    validset = []
    for room in rooms:
        for node in nodes:
            dataset = torchvision.datasets.ImageFolder(datapath + room +'-%d' % node,
                                                       transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(0.5,
                                                                                0.5)
                                                       ]))
            total_num = len(dataset)
            train_num = int(np.floor(train_part * total_num))
            train_data, valid_data = torch.utils.data.random_split(dataset=dataset,
                                                                   lengths=[train_num, total_num - train_num],
                                                                   generator=torch.Generator().manual_seed(random_seed))
            if trainset == [] and validset == []:
                trainset = train_data
                validset = valid_data
            else:
                trainset = torch.utils.data.ConcatDataset([trainset, train_data])
                validset = torch.utils.data.ConcatDataset([validset, valid_data])
    if world_size != None and rank != None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(validset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=0,
                                                   sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(validset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=0,
                                                   sampler=valid_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0,
                                                   )
        valid_loader = torch.utils.data.DataLoader(validset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0,
                                                   )

    return train_loader, valid_loader