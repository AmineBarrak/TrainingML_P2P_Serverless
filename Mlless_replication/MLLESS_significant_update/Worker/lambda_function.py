import json
import os
import pika
import ssl

# Ensure the external libraries are included in your Lambda deployment package
from utils.worker import worker
from utils.storage_backends import S3Backend

def lambda_handler(event, context):
    # Extract arguments from the event object
    worker_id = event['worker_id']
    n_workers = event['n_workers']
    broker_url = event['broker_url']
    
    credentials = pika.PlainCredentials("******", "******")

    # Configure SSL context
    ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Define worker parameters
    worker_params = {
        'dataset': 'cifar10',
        'worker_id': worker_id,
        'model': 'mobilenet',
        'executor_id': 'local',
        'asp_threshold': 0.0,
        'n_minibatches': 92,
        'bucket': 'cifar10-mlless',  # Replace with your actual bucket name
        'end_threshold': 0.0,
        'n_workers': n_workers,
        'seed': 8,
        'local': True,
        'dataset_path': None,
        'backend': S3Backend,
        'slack': 0,
        'rabbitmq_params': pika.ConnectionParameters(host=broker_url,
                                       virtual_host='/',
                                       credentials=credentials, ssl_options=pika.SSLOptions(ssl_context)),
        'n_redis': 1,
        'redis_hosts': ['localhost'],
    }

    # Execute worker with provided parameters
    worker(worker_params)

    # Return a success response
    return {
        'statusCode': 200,
        'body': json.dumps('Worker execution completed successfully')
    }


#{
#  "worker_id": 0,
#  "n_workers": 2,
#  "broker_url": "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"
#}
