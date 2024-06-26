import urllib
import logging
import boto3
from botocore.exceptions import ClientError
import pickle
import numpy as np
import time
'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

s3 = boto3.resource('s3', aws_access_key_id='***********', aws_secret_access_key='**********', region_name='us-east-1')

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
    s3 = boto3.client('s3',aws_access_key_id='***************',
                    aws_secret_access_key='**************',
                    region_name='us-east-1')
    download_path = '/tmp/{}'.format(dest_file_key)
    try:
        s3.download_file(dest_bucket_name, dest_file_key, download_path)
    except ClientError as e:
        # AllAccessDisabled error == bucket not found
        # NoSuchKey or InvalidRequest error == (dest bucket/obj == src bucket/obj)
        logging.error(e)
        return False

    return download_path
    
class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number 
         
def merge_all_workers_without_sync(bucket_name, num_workers, prefix):
    num_files = 0
    # merged_value = np.zeros(dshape, dtype=dtype)
    merged_value = []

    objects = list_bucket_objects(bucket_name)
    if objects is not None:
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            print(f"readin the following file {file_key} from {bucket_name}")
            data_bytes = get_object(bucket_name, file_key).read()
            data = pickle.loads(data_bytes)

            for i in range(len(data)):
                if num_files == 0:
                    merged_value.append(np.zeros(data[i].shape, dtype=data[i].dtype))
                merged_value[i] = merged_value[i] + data[i]

            num_files = num_files + 1
            #delete_object(bucket_name, file_key)

    # average weights
    if prefix == 'w_':
        merged_value = [value / float(num_workers) for value in merged_value]

    return merged_value
    
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
    s3 = boto3.client('s3',aws_access_key_id='***************', aws_secret_access_key='**************', region_name='us-east-1')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
    except ClientError as e:
        # AllAccessDisabled error == bucket or object not found
        logging.error(e)
        return None
    # Return an open StreamingBody object
    return response['Body']

def put_object(bucket_name, key, data):
    # Use the existing s3 resource object
    bucket = s3.Bucket(bucket_name)
    bucket.put_object(Key=key, Body=data)
    
    
def put_merged(bucket_name, merged_value, prefix, file_postfix):
    # print('put merged weight {} to bucket {}'.format(w_prefix + file_postfix, bucket_name))
    put_object(bucket_name, prefix + file_postfix, pickle.dumps(merged_value))
    
    
def list_bucket_objects(bucket_name):
    """List the objects in an Amazon S3 bucket

    :param bucket_name: string
    :return: List of bucket objects. If error, return None.
    """

    # Retrieve the list of bucket objects
    s_3 = boto3.client('s3',aws_access_key_id='*****************', aws_secret_access_key='************', region_name='us-east-1')
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
# delete the merged values of the *current or older* steps
def delete_expired_merged(bucket_name, cur_epoch, cur_batch):
    objects = list_bucket_objects(bucket_name)
    if objects is not None:
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            key_splits = file_key.split("_")
            key_batch = int(key_splits[-1])
            key_epoch = int(key_splits[-2])
            if key_epoch < cur_epoch or (key_epoch == cur_epoch and key_batch < cur_batch):
                # print("delete object {} in bucket {}".format(file_key, bucket_name))
                delete_object(bucket_name, file_key)
def delete_object(bucket_name, object_name):
    """Delete an object from an S3 bucket

    :param bucket_name: string
    :param object_name: string
    :return: True if the referenced object was deleted, otherwise False
    """

    # Delete the object
    s3 = boto3.client('s3',aws_access_key_id='****************', aws_secret_access_key='*************', region_name='us-east-1')
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

def scatter_reduce(vector, tmp_bucket, merged_bucket, num_workers, myrank, postfix):

    # vector is supposed to be a 1-d numpy array: vector is all the grads
    num_all_values = vector.size
    num_values_per_worker = num_all_values // num_workers
    residue = num_all_values % num_workers
    curr_epoch = postfix.split("_")[0]
    curr_batch = postfix.split("_")[1]

    my_offset = (num_values_per_worker * myrank) + min(residue, myrank)
    my_length = num_values_per_worker + (1 if myrank < residue else 0)
    my_chunk = vector[my_offset : my_offset + my_length]
    time_start_sync =  time.time()
    # write partitioned vector to the shared memory, except the chunk charged by myself
    for i in range(num_workers):
        if i != myrank:
            offset = (num_values_per_worker * i) + min(residue, i)
            length = num_values_per_worker + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            key = "{}_{}".format(i, myrank)
            # format of key in tmp-bucket: chunkID_workerID_epoch_batch
            put_object(tmp_bucket, key + '_' + postfix, vector[offset : offset + length].tobytes())

    # read and aggergate the corresponding chunk
    num_files = 0
    while num_files < num_workers - 1:
        objects = list_bucket_objects(tmp_bucket)
        if objects is not None:
            for obj in objects:

                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                key_splits = file_key.split("_")

                # if it's the chunk I care and it is from the current step
                 # format of key in tmp-bucket: chunkID_workerID_epoch_batch
                if key_splits[0] == str(myrank) and key_splits[2] == curr_epoch and key_splits[3] == curr_batch:

                    data = get_object(tmp_bucket, file_key).read()
                    bytes_data = np.frombuffer(data, dtype=vector.dtype)
                    my_chunk = my_chunk + bytes_data
                    num_files += 1
                    delete_object(tmp_bucket, file_key)

    # write the aggregated chunk back
    # key format in merged_bucket: chunkID_epoch_batch
    put_object(merged_bucket, str(myrank) + '_' + postfix, my_chunk.tobytes())
    time_finish_sync = time.time()


    # read other aggregated chunks
    merged_value = {}
    merged_value[myrank] = my_chunk

    num_merged_files = 0
    already_read = []
    while num_merged_files < num_workers - 1:
        objects = list_bucket_objects(merged_bucket)
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                key_splits = file_key.split("_")
                #key format in merged_bucket: chunkID_epoch_batch
                if key_splits[0] != str(myrank) and key_splits[1] == curr_epoch and key_splits[2] == curr_batch and file_key not in already_read:
                # if not file_key.startswith(str(myrank)) and file_key not in already_read:
                    # key_splits = file_key.split("_")
                    data = get_object(merged_bucket, file_key).read()
                    bytes_data = np.frombuffer(data, dtype=vector.dtype)

                    merged_value[int(key_splits[0])] = bytes_data

                    already_read.append(file_key)
                    num_merged_files += 1
    
    #time_all_sync = time_finish_sync - time_start_sync
    # reconstruct the whole vector
    result = merged_value[0]
    for k in range(1, num_workers):
        result = np.concatenate((result, merged_value[k]))
        # elif k == myrank:
        #     result = np.concatenate((result, my_chunk))
        # else:
        #     result = np.concatenate((result, merged_value[k]))
    return result, time_finish_sync
    
    
def get_merged(bucket_name, prefix, file_postfix):
    # print('get merged weight {} in bucket {}'.format(w_prefix + file_postfix, bucket_name))
    merged_value = get_object_or_wait(bucket_name, prefix + file_postfix, 0.1).read()
    merged_value_np = pickle.loads(merged_value)
    # merged_value_np = np.frombuffer(merged_value, dtype=dtype).reshape(dshape)

    return merged_value_np
    
    
    
def get_object_or_wait(bucket_name, object_name, sleep_time):
    """Retrieve an object from an Amazon S3 bucket

    :param bucket_name: string
    :param object_name: string
    :param sleep_time: float
    :return: botocore.response.StreamingBody object. If error, return None.
    """

    # Retrieve the object
    s3 = boto3.client('s3',aws_access_key_id='***********', aws_secret_access_key='***************', region_name='us-east-1')

    while True:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_name)
            # Return an open StreamingBody object
            return response['Body']
        except ClientError as e:
            # AllAccessDisabled error == bucket or object not found
            time.sleep(sleep_time)
