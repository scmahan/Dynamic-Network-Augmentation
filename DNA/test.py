import torch
import random
import os
import argparse
import datasets
from DNA import DNA
from subpolicies import sub_policies as splist
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
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
    set_seed(args.seed)
    
    dataset = datasets.get_data(args.data, args.dataroot, train=False, seed=args.seed)
    testloader = DataLoader(dataset, batch_size=args.batch_size)
       
    sub_policies = random.sample(splist, args.num_policies)
    model = DNA(args.model_name, sub_policies, num_targets=datasets.num_class(args.data))
    model.set_mode("test")
    fname = "/model_" + args.olddata + "_seed" + str(args.seed) + \
        "_epochs" + str(args.epochs_searched) + "+" + str(args.epochs_trained) + ".pt"
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Using Cuda")
        args.dir = args.dir + "-gpu"
    
    ## set save location ##
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    if use_cuda:
        model.load_state_dict(torch.load(args.dir + fname))
    else:
        model.load_state_dict(torch.load(args.dir + fname, map_location=torch.device('cpu')))
    
    log = []
    # record accuracy
    total = 0
    correct = 0
    C = np.zeros([datasets.num_class(args.data), datasets.num_class(args.data)])
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(testloader):
            if use_cuda:
                x, t = x.cuda(), t.cuda()
            
            y = model(x)

            total += t.shape[0]
            
            _, prediction = torch.max(y.data, 1)
            correct += (prediction == t).sum().item()
            
            C = C + confusion_matrix(t, prediction, labels=range(datasets.num_class(args.data)))
            
            print("Batch:",i+1,"out of",int(np.ceil(dataset.__len__()/args.batch_size)))
            
    acc = (correct/total)*100
    log.append(np.array(acc).item())
    print("Accuracy: {}".format(acc))
    print("Error: {}".format(100-acc))
    print(C)
    
    df = pd.DataFrame(log)
    df.to_pickle(args.dir + "/acc_" + args.data + "_seed" + str(args.seed) + \
        "_epochs" + str(args.epochs_searched) + "+" + str(args.epochs_trained) + ".pkl")
 
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="mnist augerino")

    parser.add_argument(
        "--dir",
        type=str,
        default='saved-outputs',
        help="training directory (default: None)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--epochs_searched",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs already searched (default: 20)",
    )
    
    parser.add_argument(
        "--epochs_trained",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs already trained (default: 200)",
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
        default="CIFAR10",
        help="which dataset to use (default: CIFAR10)"
    )
    
    parser.add_argument(
        "--olddata",
        type=str,
        default="CIFAR10",
        help="which dataset was used for previous searching/training"
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
        