import torch
import random
import os
import argparse
import datasets
import time
from DNA import DNA
from torch.utils.data import DataLoader
from subpolicies import sub_policies as splist
import pandas as pd
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

def main(args):
    start_time = time.time()
    set_seed(args.seed)
    
    dataset = datasets.get_data(args.data, args.dataroot, seed=args.seed)
    trainloader = DataLoader(dataset, batch_size=args.batch_size)
    
    sub_policies = random.sample(splist, args.num_policies)
    model = DNA(args.model_name, sub_policies, num_targets=datasets.num_class(args.data))
    model.set_mode("search")
    criterion = torch.nn.CrossEntropyLoss()
    fname = "/model_" + args.data + "_seed" + str(args.seed) + "_epochs0+0" + ".pt"
    
    opt1 = torch.optim.Adam(model.augnet.parameters(), lr=args.alr,
                            betas=(0.5,0.999), weight_decay=0.)
    opt2 = torch.optim.SGD(model.classnet.parameters(), lr=args.lr,
                           momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, float(args.epochs))
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        print("Using Cuda")
        args.dir = args.dir + "-gpu" 

    ## save init model ##
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    torch.save(model.state_dict(), args.dir + fname)

    log = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        batches = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            opt1.zero_grad()
            opt2.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt1.step()
            opt2.step()
            epoch_loss += loss.detach().item()
            batches += 1
            print("Epoch:",epoch+1,"Loss:",round(loss.item(),3))
            print("Batch:",i+1,"out of",int(np.ceil(dataset.__len__()/args.batch_size)))
            log.append(loss.item())
        scheduler.step()

    end_time = time.time()
    total_time = (end_time - start_time)/3600.
    log.append(total_time)

    df = pd.DataFrame(log)
    fname = "/model_" + args.data + "_seed" + str(args.seed) + "_epochs" + str(args.epochs) + "+0.pt"
    df.to_pickle(args.dir + "/log_" + args.data + "_seed" + str(args.seed) + "_epochs" + str(args.epochs) + "+0.pkl")
    torch.save(model.state_dict(), args.dir + fname)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DNA")

    parser.add_argument(
        "--dir",
        type=str,
        default='saved-outputs',
        help="training directory (default: None)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    
    parser.add_argument(
        "--alr",
        type=float,
        default=0.005,
        metavar="ALR",
        help="augmentation learning rate (default: 0.005)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    
    parser.add_argument(
        "--wd",
        type=float,
        default=2e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="momentum",
        help="momentum for SGD optimizer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to search (default: 20)",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="wresnet40_2",
        help="which model to use"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="ReducedCIFAR10",
        help="which dataset to use"
    )
    
    parser.add_argument(
        "--dataroot",
        type=str,
        default="~/datasets/",
        help="where dataset is located"
    )
    
    parser.add_argument(
        "--num_policies",
        type=int,
        default=105,
        help="how many policies to use (default: 105)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducibility"
    )
    
    args = parser.parse_args()

    main(args)
