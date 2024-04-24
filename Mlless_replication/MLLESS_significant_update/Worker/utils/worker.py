import time

import numpy as np
import pika 
import boto3
from .communication import RabbitMQCommunicator, BackendRabbitMQCommunicator
from .storage_S3 import StorageIterator
from .storage_backends import S3Backend
from .ModelManager import ModelPytorch,get_model
import ssl
import os 


credentials = pika.PlainCredentials("***********", "***********")
broker_url = "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"
s3 = boto3.resource('s3', aws_access_key_id='***********', aws_secret_access_key='***********', region_name='us-east-1')


def log_print(worker_index, *args, **kwargs):
    BUCKET_NAME = 'stats-mlless'
    KEY = f'mlless_worker_{worker_index}_log.txt'
    
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
        
        

def test(epoch, net, testloader, criterion, device, worker_id):
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

    avg_loss = test_loss / len(testloader.dataset)
    acc = 100. * correct / total
    log_print(worker_id, "Epoch: {}, Test Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch, avg_loss, acc))

    # Implement checkpointing here if needed

    return avg_loss, acc  # Optionally return metrics


def worker(args):
 
    t0 = time.time()
    np.seterr(all='raise')
    # basic parameters
    bucket = args['bucket']
    dataset = args['dataset']
    worker_id = args['worker_id']
    n_minibatches = args['n_minibatches']
    n_workers = args['n_workers']
    seed = args['seed']
    local = args['local']
    executor_id = args['executor_id']
    file_path = f'metrics_file_{worker_id}.txt'
    #rabbitmq_params = pika.ConnectionParameters(
    #host= 'localhost',
    #port=5672,
    # credentials=pika.PlainCredentials('guest', 'guest')
#)
    #print(f"number of workers {n_workers} and minibatches {n_minibatches}")
    
    ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    rabbitmq_params = pika.ConnectionParameters(host=broker_url,
                                       virtual_host='/',
                                       credentials=credentials,ssl_options=pika.SSLOptions(ssl_context))
                                       
    slack = args['slack']
    n_redis = 1
    
    #redis_hosts = "localhost"
    #redis_port = 6379

    iter_per_epoch = int(n_minibatches / n_workers)
    if n_minibatches % n_workers != 0:
        iter_per_epoch += 1

    # RabbitMQ & storage iterator
    #print("worker")
    storage = S3Backend(bucket_name='mlless2',
                    aws_access_key_id='AKIA2XFJ5RUTTHIUMQ7P',
                    aws_secret_access_key='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',
                    region_name='us-east-1')  # Specify the correct AWS region    
    # storage =  RedisBackend(redis_hosts=['localhost'], redis_port=6379)
    

    communicator = BackendRabbitMQCommunicator(executor_id, worker_id, slack, storage, rabbitmq_params)
  
    iterator = StorageIterator( bucket, dataset, worker_id, n_minibatches, seed, n_workers)
    total_elements = iterator.n_minibatches
    #print(f'Running worker with id={worker_id}, execution={executor_id}, nbr elements= {total_elements}')
    
    model_str = args.pop('model', None)
    args['communicator'] = communicator
    model = get_model(dataset, model_str)


    step = 0
    epoch = 0
    batch_idx = 0
    train_loss = 0
    total = 0
    correct = 0


    local_times = []
    start_all_batches = time.time()

    for minibatch in iterator:
        
        
        step_start_time = time.time()
        
        t_fetch_minibatch = iterator.cos_time
        
        log_print(worker_id, "W{} fetch dataset {} s".format(worker_id, t_fetch_minibatch))

        # Step
        ti0 = time.time()
         
        gradients_time_start = time.time()
        loss , gradients, outputs, labels = model.step(epoch, step, minibatch)
        gradients_time = time.time() - gradients_time_start 
        log_print(worker_id, "Step: {} W{} compute grads cost {} s".format(batch_idx, worker_id, gradients_time))


        t_process = time.time() - ti0

        ts0 = time.time()
        if n_workers >= 1:
            significant_updates = model.get_significant_updates(gradients)
        else:
            significant_updates = None
        t_generate_updates = time.time() - ts0 


        

        # Communicate significant updates
        t_up_0 = time.time()
        update_available = significant_updates is not None


        if update_available:
            
            communicator.send_updates(step, significant_updates)
            #print("send")

        t_write_update = time.time() - t_up_0


        # Listen to updates or supervisor msgs
        t_r_u0 = time.time()
        communicator.send_step_end(step, loss, update_available)

        iterator.n_workers = communicator.listen(n_workers, step, update_available, model)

        #print("start aggregation") 
        synchronization_time = time.time() -  t_r_u0 
        log_print(worker_id, "Step: {} W{} listen until receive expected update cost {} s".format(batch_idx, worker_id, time.time() - t_up_0))

        start_aggregation = time.time()
        aggregation_time, tmp_model_update = communicator.aggregate_updates(storage, model,file_path)
        #aggregation_time = time.time() - start_aggregation 

        log_print(worker_id, "Step: {} W{} aggregation cost {} s".format(batch_idx, worker_id, aggregation_time))
        log_print(worker_id, "Step: {} W{} synchronisation cost {} s".format(batch_idx, worker_id, time.time() - t_up_0))
        log_print(worker_id, "Step: {} W{} model update cost {} s".format(batch_idx, worker_id, tmp_model_update))
        #print("end aggregation")

	
        t_read_updates = time.time() - t_r_u0

        t_p_u0 = time.time()

        # Send model if killed
        if communicator.killed:
            communicator.send_model_on_death(model.get_weights())
        elif len(communicator.received_models) != 0:
            #print("getting updates from killed workers")
            for m in communicator.received_models:
                model.aggregate_model(m)
            communicator.received_models = []
            iter_per_epoch = int(n_minibatches / n_workers)
            if n_minibatches % n_workers != 0:
                iter_per_epoch += 1
        t_process_updates = time.time() - t_p_u0

        # Generate time stats
        step_total_time = time.time() - step_start_time
        t_read_redis_updates = communicator.fetch_update_time
        communicator.fetch_update_time = 0.0
        t_read_rabbit_updates = t_read_updates - t_read_redis_updates

        local_times.append(
            (t_fetch_minibatch, t_process, t_generate_updates,
             t_write_update, t_read_updates, t_read_redis_updates, t_read_rabbit_updates,
             t_process_updates, step_total_time))
             
        # Update training loss, total, and correct counts
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print training loss and accuracy
        
        log_print(worker_id, 'Worker: {}, Epoch: {}, Step: {}, Time: {}, Loss: {:.6f}, Loss data: {:.6f}, Accuracy: {:.2f}%'.format(worker_id, epoch + 1, batch_idx + 1, time.time() - step_total_time, train_loss / (batch_idx + 1), loss.data, 100. * correct / total))

        if step % iter_per_epoch == 0 and step != 0:
            epoch += 1
            train_loss = 0
            total = 0
            correct = 0

        # Finish execution if end is reached (killed, training end, etc.)
        if communicator.end:
            communicator.send_model_on_finish((model.get_weights(), local_times))
            #print("program is send to kill")
            break
        step += 1
        batch_idx +=1

    total_time = time.time() - t0
    log_print(worker_id, "total_time".format(total_time,file_path))
    #print(f'Finished execution of worker with id={worker_id}. Iterations = {step}. Total time: {total_time}.\n')
    communicator.send_times(total_time)
