import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
from definitions import *
from data_preprocessing import *




# lambda setting
merged_bucket = "cnn-updates2"
tmp_bucket = "cnn-grads2"
weights_prefix = 'w_'
gradients_prefix = 'g_'
local_dir = "/tmp"
# dataset setting
training_file = "training.pt"
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
    BUCKET_NAME = 'scatterreduce-results'
    KEY = f'scatter_reduce_worker_{worker_index}_log.txt'
    
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
    train_set = preprocess_cifar10(num_workers=num_worker, rank=worker_index, bucket_name=bucket, key="cifar-10-python.tar.gz", data_path='/tmp')
    log_print(worker_index, "W{} fetch dataset {} s".format(worker_index, time.time() - readS3_start))
    #print("read data cost {} s".format(time.time() - readS3_start))
    print(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # Calculate the number of batches
    num_batches = len(train_loader)

    print(f'Number of batches: {num_batches}')
    # print(enumerate(train_loader))
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
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
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):

        time_record = train(epoch, net, train_loader, optimizer, device, worker_index, num_worker, sync_mode, sync_step)
        #test(epoch, net, test_loader, device)
    put_object("time-record-s3", "time_{}".format(worker_index), pickle.dumps(time_record))


# Training
def train(epoch, net, trainloader, optimizer, device, worker_index, num_worker, sync_mode, sync_step):

    net.train()

    epoch_start = time.time()

    epoch_sync_time = 0
    num_batch = 0

    train_acc = Accuracy()
    train_loss = Average()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # print("------worker {} epoch {} batch {}------".format(worker_index, epoch+1, batch_idx+1))
        batch_start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        # get gradients and flatten it to a 1-D array
        #gradients = [param.grad.data.numpy() for param in net.parameters()]
        tmp_calculation_time = time.time()-batch_start
        log_print(worker_index, "Step: {} W{} compute grads cost {} s".format(batch_idx, worker_index, tmp_calculation_time))
        
        # print("forward and backward cost {} s".format(time.time()-batch_start))

        if sync_mode == 'model_avg':
            # apply local gradient to local model
            optimizer.step()
            # average model
            if (batch_idx+1) % sync_step == 0:
                sync_start = time.time()
                #################################reduce_broadcast####################################
                print("starting model average")
                weights = [param.data.numpy() for param in net.parameters()]
                # print("[Worker {}] Gradients before sync = {}".format(worker_index, gradients[0][0]))

                # put_object_start = time.time()
                put_object(tmp_bucket, gradients_prefix + str(worker_index), pickle.dumps(weights))
                # print("write local gradients cost {} s".format(time.time() - put_object_start))

                file_postfix = "{}_{}".format(epoch, batch_idx)
                if worker_index == 0:
                    # merge all workers
                    # merged_value_start = time.time()
                    merged_value = merge_all_workers(tmp_bucket, num_worker, gradients_prefix)
                    # print("merged_value cost {} s".format(time.time() - merged_value_start))

                    # put_merged_start = time.time()
                    # upload merged value to S3
                    put_merged(merged_bucket, merged_value, gradients_prefix, file_postfix)
                    # print("put_merged cost {} s".format(time.time() - put_merged_start))

                    # delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
                else:
                    # read_merged_start = time.time()
                    # get merged value from S3
                    merged_value = get_merged(merged_bucket, gradients_prefix, file_postfix)
                    # print("read_merged cost {} s".format(time.time() - read_merged_start))

                # print("[Worker {}] Gradients after sync = {}".format(worker_index, merged_value[0][0]))
                for layer_index, param in enumerate(net.parameters()):
                    param.data = torch.from_numpy(merged_value[layer_index])
                # gradients = [param.grad.data.numpy() for param in net.parameters()]
                # print("[Worker {}] Gradients after sync = {}".format(worker_index, gradients[0][0]))
                # print("synchronization cost {} s".format(time.time() - sync_start))
                if worker_index == 0:
                    delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
                epoch_sync_time += time.time() - sync_start

        if sync_mode == 'grad_avg':

            sync_start = time.time()

            #################################scatter_reduce####################################
            
            # print("[Worker {}] Gradients before sync = {}".format(worker_index, gradients[0][0]))
            param_dic = {}
            comm_start = time.time()
            for index, param in enumerate(net.parameters()):
                param_dic[index] = [param.grad.data.numpy().size, param.grad.data.numpy().shape]
                if index == 0:
                    flattened_param = param.grad.data.numpy().flatten()
                else:
                    flattened_param = np.concatenate((flattened_param, param.grad.data.numpy().flatten()))
            

            # merge gradients
            file_postfix = "{}_{}".format(epoch, batch_idx)
            merged_value, time_sync = scatter_reduce(flattened_param, tmp_bucket, merged_bucket, num_worker, worker_index, file_postfix)
            merged_value /= float(num_worker)
            # print("synchronisation for {} workers cost {} s".format(num_worker,time_sync))
            log_print(worker_index, "Step: {} W{} synchronisation cost {} s".format(batch_idx, worker_index, time.time() - comm_start))
            # Format the string
            formatted_string = "worker {} Synchronization for {} workers cost {} s".format(worker_index, num_worker, time_sync)
            
            update_start = time.time()


            # update the model gradients by layers
            offset = 0
            for layer_index, param in enumerate(net.parameters()):
                layer_size = param_dic[layer_index][0]
                layer_shape = param_dic[layer_index][1]
                layer_value = merged_value[offset : offset + layer_size].reshape(layer_shape)
                param.grad.data = torch.from_numpy(layer_value)
                offset += layer_size

            if worker_index == 0:
                delete_expired_merged(merged_bucket, epoch, batch_idx)
            #################################scatter_reduce####################################

            #################################reduce_broadcast####################################
            # gradients = [param.grad.data.numpy() for param in net.parameters()]
            # # print("[Worker {}] Gradients before sync = {}".format(worker_index, gradients[0][0]))

            # # put_object_start = time.time()
            # put_object(tmp_bucket, gradients_prefix + str(worker_index), pickle.dumps(gradients))
            # # print("write local gradients cost {} s".format(time.time() - put_object_start))

            # file_postfix = "{}_{}".format(epoch, batch_idx)
            # if worker_index == 0:
            #     # merge all workers
            #     # merged_value_start = time.time()
            #     merged_value = merge_all_workers(tmp_bucket, num_worker, gradients_prefix)
            #     # print("merged_value cost {} s".format(time.time() - merged_value_start))

            #     # put_merged_start = time.time()
            #     # upload merged value to S3
            #     put_merged(merged_bucket, merged_value, gradients_prefix, file_postfix)
            #     # print("put_merged cost {} s".format(time.time() - put_merged_start))
            #     # delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)

            # else:
            #     # read_merged_start = time.time()
            #     # get merged value from S3
            #     merged_value = get_merged(merged_bucket, gradients_prefix, file_postfix)
            #     # print("read_merged cost {} s".format(time.time() - read_merged_start))

            # # print("[Worker {}] Gradients after sync = {}".format(worker_index, merged_value[0][0]))
            # for layer_index, param in enumerate(net.parameters()):
            #     param.grad.data = torch.from_numpy(merged_value[layer_index])
            # gradients = [param.grad.data.numpy() for param in net.parameters()]
            # print("[Worker {}] Gradients after sync = {}".format(worker_index, gradients[0][0]))
            # # print("synchronization cost {} s".format(time.time() - sync_start))

            # if worker_index == 0:
            #     delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
            #################################reduce_broadcast####################################
            
            optimiser = time.time()
            epoch_sync_time += time.time() - sync_start
            tmp_sync_time2 = time.time() - sync_start
            log_print(worker_index, "compare synchronization cost {} s".format(tmp_sync_time2))
            optimizer.step()
            
            log_print(worker_index, "Step: {} W{} model update cost {} s".format(batch_idx, worker_index, time.time() - update_start))
            
            log_print(worker_index, "Step: {} W{} optimiser cost {} s".format(batch_idx, worker_index, time.time() - optimiser))

        if sync_mode == 'cen':
            optimizer.step()

        train_acc.update(outputs, targets)
        train_loss.update(loss.item(), inputs.size(0))
        log_print(worker_index, 'Worker: {}, Epoch: {}, Step: {}, time: {}, Loss:{}'.format(worker_index, epoch+1, batch_idx+1, time.time() - batch_start, loss.data))

        if num_batch % 10 == 0:
            log_print(worker_index, "Epoch {} Batch {} training Loss:{}, Acc:{}".format(epoch+1, num_batch, train_loss, train_acc))
        num_batch += 1

    epoch_time = time.time() - epoch_start
    log_print(worker_index, "Epoch {} has {} batches, time = {} s, sync time = {} s, cal time = {} s"
          .format(epoch+1, num_batch, epoch_time, epoch_sync_time, epoch_time - epoch_sync_time))

    return train_loss, train_acc


def test(epoch, net, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    #avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    log_print(worker_index, "Accuracy of epoch {} on test set is {}".format(epoch, accuracy))





#    event_data = {
#        "data_bucket": "cld202-sagemaker",
#        "rank": rank,  # Worker index for this process
#        "num_workers": num_workers  # Total number of workers
#    }

