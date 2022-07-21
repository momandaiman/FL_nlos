#!/usr/local/ancoonda3/envs/nas/bin/python
import collections
import time
import glob
import sys
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import numpy as np

from DataProcess import DataLoader
from utils import AvgrageMeter, accuracy, create_exp_dir, Logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(*list(models.resnet18().children())[0:-1])
        self.lstm1 = nn.LSTM(input_size=512 * 1 * 1, hidden_size=256, num_layers=1, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256 * 2, hidden_size=16, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(16 * 2, 2)

    def forward(self, x, ):
        x = self.cnn(x)
        batch_size = x.shape[0]
        x = x.view(1, batch_size, 512 * 1 * 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        x = x.squeeze(0)
        return x

def main():
    parser = argparse.ArgumentParser()
    # Distributed parameters
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    # training parameters
    parser.add_argument('--client_num', default=4, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    # Federal learning
    parser.add_argument('--federal_rounds', default=100, type=int)
    parser.add_argument('--mix_para', default=0.9, type=float)
    parser.add_argument('--sigma', default=0, type=float)
    # dataset
    parser.add_argument('--datapath', default='/home/while1training/DATA/NLOS/', type=str)
    parser.add_argument('--train_part', default=0.8, type=float)
    # saving
    parser.add_argument('--save', default='EXP', type=str)
    args = parser.parse_args()


    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logging.info(eachArg + ' ' + str(value))

    msg_buffer = mp.Manager().Queue()

    # Distributed training
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2333'
    mp.spawn(federal_train_and_test, nprocs=args.gpus, args=(args, msg_buffer))


def federal_train_and_test(gpu, args, msg_buffer):
    rank = args.gpus * args.nr + gpu
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)

    # sys.stdout = Logger(os.path.join(args.save, 'log.txt'))
    if gpu == 0:
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    model = Net().cuda(gpu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Wrap the model
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model = DDP(model)

    client_weights = [model.module.state_dict() for client in range(args.client_num)]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.federal_rounds))

    # Dataset
    train_loader = []
    rooms = ['restroom', '526', '511', 'room4']
    for room in rooms:
        train_loader.append(DataLoader(rooms=[room],
                                       nodes=[1, 2, 3, 4],
                                       train_part=args.train_part,
                                       world_size=args.world_size,
                                       rank=rank,
                                       batch_size=args.batch_size,
                                       )[0])
    valid_loader = DataLoader(rooms=rooms,
                              nodes=[1, 2, 3, 4],
                              train_part=args.train_part,
                              world_size=args.world_size,
                              rank=rank,
                              batch_size=args.batch_size,
                              )[1]

    for federal_rounds in range(args.federal_rounds):
        # Training
        data_num = [0 for client in range(args.client_num)]
        for client in range(args.client_num):
            start_time = time.time()
            data_num[client] = 0
            if federal_rounds != 0:
                weight_keys = list(client_weights[0].keys())
                for key in weight_keys:
                    client_weights[client][key] = client_weights[client][key] * args.mix_para + \
                                                  server_weights[key] * (1 - args.mix_para)

            model.module.load_state_dict(client_weights[client])

            model.train()
            train_objs = AvgrageMeter()
            train_top1 = AvgrageMeter()
            train_top2 = AvgrageMeter()

            for epoch in range(args.epochs):

                for i, (images, labels) in enumerate(train_loader[client]):
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    optimizer.step()
                    train_prec1, train_prec2 = accuracy(outputs, labels, topk=(1, 2))
                    n = images.size(0)
                    train_objs.update(scaled_loss.item(), n)
                    train_top1.update(train_prec1.item(), n)
                    train_top2.update(train_prec2.item(), n)
                    if epoch == 0:
                        data_num[client] = data_num[client] + n

            msg_buffer.put('GPU {} Federal rounds [{}/{}] Client {} train_acc {:.4f} train_loss {:.4f} learning_rate {:.6f} Time {}'.format(
                    gpu,
                    federal_rounds + 1,
                    args.federal_rounds,
                    client,
                    train_top1.avg,
                    loss.item(),
                    scheduler.get_lr()[0],
                    time.time() - start_time
                ))
            if gpu == 0:
                for i in range(args.gpus):
                    msg = msg_buffer.get()
                    logging.info(msg)
                    del msg
            client_weights[client] = model.module.state_dict()
        scheduler.step()

        # Update server_weights
        weight_keys = list(client_weights[0].keys())
        server_weights = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for client in range(args.client_num):
                key_sum = key_sum + client_weights[client][key] * data_num[client] / sum(data_num) * np.random.normal(loc=1, scale=np.sqrt(args.sigma))
            server_weights[key] = key_sum

        # Validation
        start_time = time.time()
        model.module.load_state_dict(server_weights)
        model.eval()
        test_objs = AvgrageMeter()
        test_top1 = AvgrageMeter()
        test_top2 = AvgrageMeter()
        for i, (images, labels) in enumerate(valid_loader):
            with torch.no_grad():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_prec1, test_prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = images.size(0)
            test_objs.update(loss.item(), n)
            test_top1.update(test_prec1.item(), n)
            test_top2.update(test_prec2.item(), n)

        msg_buffer.put(
            'GPU {} valid_acc {:.4f} valid_loss {:.4f} Time {}'.format(
                gpu,
                test_top1.avg,
                loss.item(),
                time.time() - start_time
            ))
        if gpu == 0:
            for i in range(args.gpus):
                msg = msg_buffer.get()
                logging.info(msg)
                del msg
    # save
    if gpu == 1:
        for client in range(args.client_num):
            torch.save(client_weights[client], os.path.join(args.save, 'client_weights_%d.pt' % client))
        torch.save(server_weights, os.path.join(args.save, 'server_weights.pt'))


if __name__ == '__main__':
    main()
