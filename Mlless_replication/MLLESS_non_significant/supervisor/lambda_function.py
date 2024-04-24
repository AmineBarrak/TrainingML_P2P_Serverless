import json
import time
import ssl
import pika
import os

# Assuming utils is a package containing your supervisor and RedisBackend modules
from utils.supervisor import supervisor



def lambda_handler(event, context):
    # Extract parameters from the event object
    n_workers =  event['n_workers']
    broker_url =  event['broker_url']

    # Define the credentials for RabbitMQ
    credentials = pika.PlainCredentials("*********", "************")

    def run_supervisor(params):
        ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        rabbitmq_params = pika.ConnectionParameters(
            host=broker_url,
            virtual_host='/',
            credentials=credentials,
            ssl_options=pika.SSLOptions(ssl_context)
        )

        # Attempt to connect to RabbitMQ
        while True:
            try:
                connection = pika.BlockingConnection(rabbitmq_params)
                break
            except pika.exceptions.ConnectionClosed as exception:
                print(f"There was an error initializing RabbitMQ - {exception}. Trying again in 5 seconds...")
                time.sleep(5)

        channel = connection.channel()
        executor_id = 'local'
        channel.exchange_declare(exchange=executor_id, exchange_type='fanout')

        # Declare queues for supervisor, results, and client
        channel.queue_declare('{}_supervisor'.format(executor_id))
        channel.queue_declare('{}_results'.format(executor_id))
        channel.queue_declare('{}_client'.format(executor_id))

        # Declare worker queues and bind them
        for w in range(n_workers):
            queue = '{}_w{}'.format(executor_id, w)
            channel.queue_declare(queue=queue)
            channel.queue_bind(queue=queue, exchange=executor_id)

        # Start the supervisor with the provided parameters
        supervisor(**params)

    # Supervisor parameters
    sup_params = {
        'bucket': 'cifar10-mlless',
        'executor_id': 'local',
        'n_workers': n_workers,
        'threshold': 0.0,
        'epochs': 2000,  # n+1 epochs, counting starts from 0
        'n_minibatches': 92,
        'remove_threshold': None,
        'remove_interval': None,
        'max_time': None,
        'slack': 0,
        'n_redis': 1,
        'redis_hosts': ['localhost'],
    }

    # Run the supervisor function with the parameters
    run_supervisor(sup_params)

    # Return a response, Lambda handler needs to return something
    return {
        'statusCode': 200,
        'body': json.dumps('Supervisor run successfully')
    }


#{
#  "n_workers": 2,
#  "broker_url": "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"
#}
