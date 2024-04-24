import json


import sys
import time
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import pika
import ssl
import pickle
import os
from definitions import *
from synchronisation import *
from data_preprocessing import *


s3 = boto3.resource('s3', aws_access_key_id='***********', aws_secret_access_key='************', region_name='us-east-1')
credentials = pika.PlainCredentials("******", "*********")
# Create an SSL context


# ~ if response.status_code == 200:
	  # ~ with open('AmazonRootCA1.pem', 'wb') as f:
		  # ~ f.write(response.content)
		  # ~ print("AmazonRootCA1.pem file downloaded successfully.")
# ~ else:
	  # ~ print("Failed to download the file. Status code:", response.status_code)
ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Set up connection parameters with the SSL context
connection_parameters = pika.ConnectionParameters("b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com",
													virtual_host='/',
													credentials = credentials,
													ssl_options=pika.SSLOptions(ssl_context))




connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
sync_queue = "sync_queue"
# Ensure the queue exists
channel.queue_declare(queue=sync_queue, durable=True)
	
	

# lambda setting
merged_bucket = "cnn-updates3"
tmp_bucket = "cnn-grads3"
garbage = "garbadge"
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
max_norm = 5.0
clip_value = 0.5

os.environ['AWS_ACCESS_KEY_ID'] = '**********'
os.environ['AWS_SECRET_ACCESS_KEY'] = '************'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

def log_print(worker_index, *args, **kwargs):
	BUCKET_NAME = 'spirt-results'
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
	model_start = time.time()

	net = MobileNet()


	print("Model: MobileNet")

	net = net.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
	
	model_end = time.time()
	print(f"Model: {model_end-model_start}")


	for epoch in range(num_epochs):
		time_record = train(epoch, net, train_loader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step)
	#	test(epoch, net, test_loader, criterion, device)
	#put_object("time-record-s3", "time_{}".format(worker_index), pickle.dumps(time_record))


# Training
def train(epoch, net, trainloader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step):
	
	
	
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	sync_epoch_time = []
	write_local_epoch_time = []
	calculation_epoch_time = []
	sum_of_grad = 0
	accumulate_steps = 0
	#optimizer.zero_grad()
	

	stored_gradients = []

	# Compute gradients
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		# print("------worker {} epoch {} batch {} sync mode '{}'------".format(worker_index, epoch+1, batch_idx+1,sync_mode))
		batch_start = time.time()
		#accumulate_steps += 1
		optimizer.zero_grad()


		inputs, targets = inputs.to(device), targets.to(device)

		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		
		# Store gradients as NumPy arrays
		gradients = [param.grad.data.cpu().numpy() for param in net.parameters()]
		stored_gradients.append(gradients)
		put_object(garbage, str(batch_idx) + str(epoch) + str(worker_index), pickle.dumps(gradients))
		
		
		log_print(worker_index, "i am worker {} processing batch {} cost {} ...".format(worker_index, batch_idx, time.time()-batch_start))
		
		if batch_idx == 3:
			break

	# Average the gradients
	# Duplicate objects until the count reaches 49
# 	while len(stored_gradients) < 49:
	   # stored_gradients += stored_gradients[:]

	# Truncate the list if it exceeds 49
# 	stored_gradients = stored_gradients[:49]
	
	average_start = time.time()
	averaged_gradients = [np.mean([batch_grads[i] for batch_grads in stored_gradients], axis=0) for i in range(len(stored_gradients[0]))]
	
	
	print(f"i am worker {worker_index} start synching ...")


	put_object(tmp_bucket, gradients_prefix + str(worker_index), pickle.dumps(averaged_gradients))

	average_stop = time.time()
	log_print(worker_index, "i am worker {} average gradients cost {} ...".format(worker_index, average_stop-average_start))

	# Send msg to the queue
	start_sync = time.time()
	send_message(channel=channel, queue_name= sync_queue, message=f"peer{worker_index} completed")
	sync_flag = wait_for_n_messages_or_timeout(channel=channel, queue_name=sync_queue, n= num_worker)
	sync_stop = time.time()
	log_print(worker_index, "i am worker {} queue notification and sync cost {} ...".format(worker_index, sync_stop-start_sync)) 
		
	if sync_flag:
		print("all worker are sync")
		file_postfix = "{}_{}".format(epoch, batch_idx)
		
		
		# merge all workers
		merged_value_start = time.time()
		merged_value = merge_all_workers_without_sync(tmp_bucket, num_worker, gradients_prefix)
		
		
		put_object(garbage, "merge_" + str(epoch) + str(worker_index), pickle.dumps(merged_value))
		
		put_merged_stop = time.time()
		
		log_print(worker_index, "i am worker {} aggregate cost {} ...".format(worker_index, put_merged_stop-merged_value_start)) 
		

		
		# upload merged value to S3 for tracability
		# put_merged(merged_bucket, merged_value, gradients_prefix, file_postfix)

		


		# Update Model
		update_model_start = time.time()
		for layer_index, param in enumerate(net.parameters()):
				param.grad = Variable(torch.from_numpy(merged_value[layer_index]))
		# Scale and set the gradients for each parameter
		#for param, accumulated_grad in zip(net.parameters(), accumulated_gradients):
		#	# Ensure the gradient is on the correct device and scaled
		#	scaled_grad = (accumulated_grad / accumulate_steps).to(param.device)
		#	param.grad = scaled_grad	
		

		print("update with averaged grads from all workers")
	else:
		# Update Model
		print("update with local grads")
		#for layer_index, param in enumerate(net.parameters()):
		#	param.grad = Variable(torch.from_numpy(gradients[layer_index]))
			
			
		# Scale and set the gradients for each parameter
		for layer_index, param in enumerate(net.parameters()):
				param.grad = Variable(torch.from_numpy(averaged_gradients[layer_index]))
		#for param, accumulated_grad in zip(net.parameters(), accumulated_gradients):
			# Ensure the gradient is on the correct device and scaled
		#	scaled_grad = (accumulated_grad / accumulate_steps).to(param.device)
		#	param.grad = scaled_grad	

	
	
	# Clip gradients to prevent explosion
	#torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
	#print(f"i am worker {worker_index} finish clipping ...")
	
	optimizer.step()
	#optimizer.zero_grad()
	if sync_flag:
		log_print(worker_index, "i am worker {} model update cost {} ...".format(worker_index, time.time()-update_model_start)) 
	
	
	train_loss += loss.item()
	_, predicted = outputs.max(1)
	total += targets.size(0)
	correct += predicted.eq(targets).sum().item()
	
	accuracy = 100. * correct / total

	
	if worker_index == 0:
		empty_queue(channel, sync_queue)
		#delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
		print("cleanup done worker 0")
	#log_print('Worker: {}, Epoch: {}, Step: {}, sync cost: {}, Loss:{}'.format(worker_index, epoch+1, batch_idx+1, tmp_sync_time, loss.data))
	log_print(worker_index, 'Worker: {}, Epoch: {}, Step: {}, Time: {:.2f}, Loss: {:.6f}, Accuracy: {:.2f}%'.format(worker_index, epoch+1, batch_idx+1, time.time() - batch_start, loss.item(), accuracy))



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
			#	 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	# Save checkpoint.
	acc = 100.*correct/total
	log_print(worker_index, "Accuracy of epoch {} on test set is {}".format(epoch, acc))




# 	event_data = {
# 		"data_bucket": "cld202-sagemaker",
# 		"rank": rank,  # Worker index for this process
# 		"num_workers": num_workers  # Total number of workers
# 	}

