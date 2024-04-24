from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import os
import os.path
import tarfile

import random
import boto3
import time


def empty_bucket(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Iterate over all objects in the bucket and delete them
    for obj in bucket.objects.all():
        if obj.key != 'cifar-10-python.tar.gz':
            obj.delete()
            print(f"Deleted: {obj.key}")
        else:
            print(f"Skipped: {obj.key}")
            

        
        
class CIFAR10_subset(data.Dataset):

    def __init__(self, train, train_data, train_labels, test_data, test_labels, transform=None, target_transform=None):
        # self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data = train_data
            self.train_labels = train_labels
        else:
            self.test_data = test_data
            self.test_labels = test_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
            
            
            
            





s3 = boto3.resource('s3', aws_access_key_id='**************', aws_secret_access_key='**************', region_name='us-east-1')

base_folder = 'cifar-10-batches-py'
processed_folder = 'processed'


# Download files from S3 and unzip them
def preprocess_cifar10(num_workers, rank, bucket_name='cifar10dataset', key="cifar-10-python.tar.gz",
                       data_path='tmp/data' ):
    # download zipped file from S3
    print('==> Downloading data from S3..')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    s3.Bucket(bucket_name).download_file(key, os.path.join(data_path, key))

    # extract file
    cwd = os.getcwd()
    tar = tarfile.open(os.path.join(data_path, key), "r:gz")
    os.chdir(data_path)
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    # delete the zipped file
    os.remove(os.path.join(data_path, key))

    # Data
    print('==> Preprocessing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #transform_test = transforms.Compose([
    #    transforms.ToTensor(),
     #   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #])

    # training set
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    # test set
    #testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)

    # save test dataset back to S3
    #with open(os.path.join(data_path, "test.pt"), 'wb') as f:
     #   torch.save(testset, f)
    
    #test_subset_load = torch.load(os.path.join(data_path, 'test.pt'))


    # shuffle training set
    print('==> Shuffling and partitioning training data..')
    num_examples = len(trainset)
    indices = list(range(num_examples))
    random.shuffle(indices)

    num_examples_per_worker = num_examples // num_workers
    residue = num_examples % num_workers
    
    
    start = (num_examples_per_worker * rank) + min(residue, rank)
    num_examples_real = num_examples_per_worker + (1 if rank < residue else 0)
    print(trainset)
    
    # print("first 10 labels of original[rank]:{}".format([trainset.train_labels[i] for i in indices[start:start+num_examples_real]][0:10]))
    training_subset = CIFAR10_subset(train=True,
                                     train_data=[trainset.data[i] for i in indices[start:start+num_examples_real]],
                                     train_labels=[trainset.targets[i] for i in indices[start:start+num_examples_real]],
                                     test_data=None, test_labels=None, transform=transform_train)
    # print("first 10 labels of subset[rank]:{}".format(training_subset.train_labels[0:10]))
    
    # Save the training subset for the current worker back to S3
    with open(os.path.join(data_path, 'training_{}.pt'.format(rank)), 'wb') as f:
        torch.save(training_subset, f)
    
    # Optional: Load the subset to verify
    train_subset_load = torch.load(os.path.join(data_path, 'training_{}.pt'.format(rank)))
    
    return train_subset_load
    #, test_subset_load


if __name__ == "__main__":
    bucket_name = 'cld202-sagemaker'
    empty_bucket(bucket_name)
    preprocess_cifar10(bucket_name = bucket_name,  key = "cifar-10-python.tar.gz", data_path = '.', num_workers = 4)
