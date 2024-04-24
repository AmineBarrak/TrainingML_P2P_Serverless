from __future__ import print_function
import os
import os.path
import sys
import time
import logging
import random
import urllib
import pickle
import numpy as np

# Third-party library imports
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from botocore.exceptions import ClientError
from torch.autograd import Variable
import tarfile
import random
import json


def list_objects_in_bucket(bucket):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket)

    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("No objects found in the bucket.")




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






class MobileNet_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNet_Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()




def download_file(dest_bucket_name, dest_file_key):
    """Fetch an file to an Amazon S3 bucket

    The src_data argument must be of type bytes or a string that references
    a file specification.

    :param dest_bucket_name: string
    :param dest_file_key: string
    :return: download path if get the file successfully, otherwise
    False
    """

    # get the file
    s3 = boto3.client('s3')
    download_path = '/tmp/{}'.format(dest_file_key)
    try:
        s3.download_file(dest_bucket_name, dest_file_key, download_path)
    except ClientError as e:
        # AllAccessDisabled error == bucket not found
        # NoSuchKey or InvalidRequest error == (dest bucket/obj == src bucket/obj)
        logging.error(e)
        return False

    return download_path

def merge_all_workers(bucket_name, num_workers, prefix):
    num_files = 0
    # merged_value = np.zeros(dshape, dtype=dtype)
    merged_value = []

    while num_files < num_workers:
        objects = list_bucket_objects(bucket_name)
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                data_bytes = get_object(bucket_name, file_key).read()
                data = pickle.loads(data_bytes)

                for i in range(len(data)):
                    if num_files == 0:
                        merged_value.append(np.zeros(data[i].shape, dtype=data[i].dtype))
                    merged_value[i] = merged_value[i] + data[i]

                num_files = num_files + 1
                delete_object(bucket_name, file_key)

    # average weights
    if prefix == 'w_':
        merged_value = [value / float(num_workers) for value in merged_value]

    return merged_value

def get_object(bucket_name, object_name):
    """Retrieve an object from an Amazon S3 bucket

    :param bucket_name: string
    :param object_name: string
    :return: botocore.response.StreamingBody object. If error, return None.
    """

    # Retrieve the object
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
    except ClientError as e:
        # AllAccessDisabled error == bucket or object not found
        logging.error(e)
        return None
    # Return an open StreamingBody object
    return response['Body']

def put_merged(bucket_name, merged_value, prefix, file_postfix):
    # print('put merged weight {} to bucket {}'.format(w_prefix + file_postfix, bucket_name))
    put_object(bucket_name, prefix + file_postfix, pickle.dumps(merged_value))

def list_bucket_objects(bucket_name):
    """List the objects in an Amazon S3 bucket

    :param bucket_name: string
    :return: List of bucket objects. If error, return None.
    """

    # Retrieve the list of bucket objects
    s_3 = boto3.client('s3')
    try:
        response = s_3.list_objects_v2(Bucket=bucket_name)
    except ClientError as e:
        # AllAccessDisabled error == bucket not found
        logging.error(e)
        return None

    # Only return the contents if we found some keys
    if response['KeyCount'] > 0:
        return response['Contents']

    return None

def delete_object(bucket_name, object_name):
    """Delete an object from an S3 bucket

    :param bucket_name: string
    :param object_name: string
    :return: True if the referenced object was deleted, otherwise False
    """

    # Delete the object
    s3 = boto3.client('s3')
    try:
        s3.delete_object(Bucket=bucket_name, Key=object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True



def delete_expired(bucket_name, cur_epoch, cur_batch, prefix):
    objects = list_bucket_objects(bucket_name)
    if objects is not None:
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            if file_key.startswith(prefix):
                key_splits = file_key.split("_")
                key_batch = int(key_splits[-1])
                key_epoch = int(key_splits[-2])
                if key_epoch < cur_epoch or (key_epoch == cur_epoch and key_batch < cur_batch):
                    print("delete object {} in bucket {}".format(file_key, bucket_name))
                    delete_object(bucket_name, file_key)

def get_object_or_wait(bucket_name, object_name, sleep_time):
    """Retrieve an object from an Amazon S3 bucket

    :param bucket_name: string
    :param object_name: string
    :param sleep_time: float
    :return: botocore.response.StreamingBody object. If error, return None.
    """

    # Retrieve the object
    s3 = boto3.client('s3')

    while True:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_name)
            # Return an open StreamingBody object
            return response['Body']
        except ClientError as e:
            # AllAccessDisabled error == bucket or object not found
            time.sleep(sleep_time)

def get_merged(bucket_name, prefix, file_postfix):
    # print('get merged weight {} in bucket {}'.format(w_prefix + file_postfix, bucket_name))
    merged_value = get_object_or_wait(bucket_name, prefix + file_postfix, 0.1).read()
    merged_value_np = pickle.loads(merged_value)
    # merged_value_np = np.frombuffer(merged_value, dtype=dtype).reshape(dshape)

    return merged_value_np

"""## dataset preprocess into batches within S3 for workers"""




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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # training set
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    # test set
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)

    # save test dataset back to S3
    with open(os.path.join(data_path, "test.pt"), 'wb') as f:
        torch.save(testset, f)
    
    test_subset_load = torch.load(os.path.join(data_path, 'test.pt'))


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
    
    return train_subset_load, test_subset_load



s3 = boto3.resource('s3', aws_access_key_id='***********', aws_secret_access_key='************', region_name='us-east-1')


# lambda setting
merged_bucket = "cnn-updates"
tmp_bucket = "cnn-grads"
weights_prefix = 'w_'
gradients_prefix = 'g_'
local_dir = "/tmp"
# dataset setting
test_file = "test.pt"

# sync up mode
sync_mode = 'grad_avg'
#sync_mode = 'model_avg'
sync_step = 1

# learning algorithm setting
learning_rate = 0.01
batch_size = 256
num_epochs = 2000




def log_print(worker_index, *args, **kwargs):
    BUCKET_NAME = 'allreduce-results'
    KEY = f'all_reduce_worker_{worker_index}_log.txt'
    
    # Convert the args to a string as it would be printed
    log_message = ' '.join(map(str, args)) + '\n'
    
    # Fetch the existing log content from S3
    obj = s3.Object(BUCKET_NAME, KEY)
    
    
    try:
        response = obj.get()
        existing_log = response['Body'].read().decode('utf-8')
    except s3.meta.client.exceptions.NoSuchKey:
        # If the key does not exist, set existing_log to an empty string
        existing_log = ''
    
   

    
    # Append the new log message to the existing content
    new_log_content = existing_log + log_message

    # Write the updated log back to S3
    obj.put(Body=new_log_content.encode('utf-8'))

def put_object(bucket_name, key, data):
    # Use the existing s3 resource object
    bucket = s3.Bucket(bucket_name)
    bucket.put_object(Key=key, Body=data)
def lambda_handler(event, context):
    start_time = time.time()
    bucket = event['data_bucket']
    worker_index = event['rank']
    num_worker = event['num_workers']
    key = 'training_{}.pt'.format(worker_index)
    print('data_bucket = {}\n worker_index:{}\n num_worker:{}\n key:{}'.format(bucket, worker_index, num_worker, key))

    # read file from s3
    
    readS3_start = time.time()
    train_set, test_set = preprocess_cifar10(num_workers=num_worker, rank=worker_index, bucket_name=bucket, key="cifar-10-python.tar.gz", data_path='/tmp')
    
    
    log_print(worker_index, "W{} fetch dataset {} s".format(worker_index, time.time() - readS3_start))
    print(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print(enumerate(train_loader))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = 'cpu'
    # best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    #Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    #net = torchvision.models.resnet50()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()

    print("Model: MobileNet")

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        time_record = train(epoch, net, train_loader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step)
        test(epoch, net, test_loader, criterion, device)
    put_object("time-record-s3", "time_{}".format(worker_index), pickle.dumps(time_record))


# Training
def train(epoch, net, trainloader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    sync_epoch_time = []
    write_local_epoch_time = []
    calculation_epoch_time = []
    # print("start training")
    print(type(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print("------worker {} epoch {} batch {} sync mode '{}'------".format(worker_index, epoch+1, batch_idx+1,sync_mode))
        batch_start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        gradients = [param.grad.data.numpy() for param in net.parameters()]
        tmp_calculation_time = time.time()-batch_start
        log_print(worker_index, "Step: {} W{} compute grads cost {} s".format(batch_idx, worker_index, tmp_calculation_time))
        if batch_idx != 0:
            calculation_epoch_time.append(tmp_calculation_time)
        if sync_mode == 'grad_avg':
            sync_start = time.time()
            
            # print("[Worker {}] Gradients before sync = {}".format(worker_index, gradients[0][0]))

            put_object_start = time.time()
            put_object(tmp_bucket, gradients_prefix + str(worker_index), pickle.dumps(gradients))
            tmp_write_local_epoch_time = time.time() - put_object_start
            # print("write local gradients cost {} s".format(tmp_write_local_epoch_time))
            if batch_idx !=0 :
                write_local_epoch_time.append(tmp_write_local_epoch_time)
            file_postfix = "{}_{}".format(epoch, batch_idx)
            if worker_index == 0:
                # merge all workers
                merged_value_start = time.time()
                merged_value = merge_all_workers(tmp_bucket, num_worker, gradients_prefix)
                # print("merged_value cost {} s".format(time.time() - merged_value_start))


                # print("size of gradients = {}".format(sys.getsizeof(pickle.dumps(merged_value))/1024/1024))

                put_merged_start = time.time()
                # upload merged value to S3
                put_merged(merged_bucket, merged_value, gradients_prefix, file_postfix)
                # print("put_merged cost {} s".format(time.time() - put_merged_start))
                delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
                
            else:

                
                # get merged value from redis
                merged_value = get_merged(merged_bucket, gradients_prefix, file_postfix)
                # print("read_merged cost {} s".format(time.time() - read_merged_start))
            
            tmp_sync_time = time.time() - sync_start
            log_print(worker_index, "Step: {} W{} synchronisation cost {} s".format(batch_idx, worker_index, tmp_sync_time))
            start_model_update = time.time()
            for layer_index, param in enumerate(net.parameters()):
                param.grad = Variable(torch.from_numpy(merged_value[layer_index]))

            tmp_sync_time2 = time.time() - sync_start
            log_print(worker_index, "compare synchronization cost {} s".format(tmp_sync_time2))
            if worker_index != 0:
                tmp_model_update = time.time() - start_model_update
                log_print(worker_index, "Step: {} W{} model update cost {} s".format(batch_idx, worker_index, tmp_model_update))
            # print("synchronization cost {} s".format(tmp_sync_time))
            if batch_idx != 0:
                sync_epoch_time.append(tmp_sync_time)

            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        # print("batch cost {} s".format(time.time() - batch_start))
        # if (batch_idx + 1) % 10 == 0:
                    # Format the string
        #time_sync = finish_sync - put_object_start
        #formatted_string = "worker {} Synchronization for {} workers cost {} s".format(worker_index, num_worker, time_sync)

        # Write the formatted string to a file
        #with open('log_'+str(num_worker)+'workers.txt', 'a') as file:
        #    file.write(formatted_string + '\n')  # Adding a newline character for readability
        log_print(worker_index, 'Worker: {}, Epoch: {}, Step: {}, time: {}, Loss:{}'.format(worker_index, epoch+1, batch_idx+1, time.time() - batch_start, loss.data))
        log_print(worker_index, 'Worker: {}, Epoch: {}, Step: {}, Time: {}, Loss: {}, Accuracy: {}'.format(worker_index, epoch+1, batch_idx+1, time.time() - batch_start, loss.data, accuracy))
    return sync_epoch_time, write_local_epoch_time, calculation_epoch_time

def test(epoch, net, testloader, criterion, device):
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    log_print(worker_index, "Accuracy of epoch {} on test set is {}".format(epoch, acc))


#    event_data = {
#        "data_bucket": "cld202-sagemaker",
#        "rank": rank,  # Worker index for this process
#        "num_workers": num_workers  # Total number of workers
#    }



