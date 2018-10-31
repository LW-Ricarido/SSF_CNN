from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10,MNIST
from torch.utils.data import DataLoader
from datasets import SmallNORB
import os
import argparse
parser = argparse.ArgumentParser(description='Parser for all the training options')
parser.add_argument('--dataset',type=str)
parser.add_argument('--size',type=int)
parser.add_argument('--data_dir',type=str)
parser.add_argument('--batch_size',type=int)
args = parser.parse_args()
def get_train_datasets(args):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.size,args.size)),
        transforms.ToTensor(),
        transforms.ToPILImage(),
    ])

    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(root=args.data_dir,train=True,
                          transform=transforms.Compose([
                              transforms.Resize(args.size,args.size),
                              transforms.ToTensor(),
                              transforms.ToPILImage()
                          ]))
    elif args.dataset == 'MNIST':
        dataset = MNIST(root=args.data_dir,train=True,transform=transform)
    else:
        dataset = SmallNORB(root=args.data_dir,train=True,transform=transform)
    return dataset

if __name__ == '__main__':
    args.dataset = 'MNIST'
    args.size = 224
    args.data_dir = '../DataSets/MNIST'
    args.batch_size = 1
    dataset = get_train_datasets(args)
    train_set = open(os.path.join("../DataSets/MNISTimg/train.txt"),'w')
    trans = transforms.ToPILImage()
    for i  in range(60000):
        img ,target = dataset.__getitem__(i)
        train_set.write(str(i)+'.bmp '+ str(target)+'\n')
        img.save(os.path.join("../DataSets/MNISTimg/"+str(i)+'.bmp'))


