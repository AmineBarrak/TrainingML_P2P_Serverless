import time
import numpy as np 
import pika 
import ssl 
import os 
from .communication import SupervisorCommunicator
from .storage_backends import StorageBackend 
from .storage_backends import S3Backend



credentials = pika.PlainCredentials("*********", "************")
broker_url = "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"
# take into account only current step, though save slack steps ahead.
def supervisor( bucket, executor_id, n_workers, threshold, epochs, n_minibatches, remove_threshold, remove_interval, max_time, slack, n_redis, redis_hosts):
    init_time = time.time()
    msg = f"[supervisor]: Started supervisor execution {executor_id}. Supervising {n_workers} parallel workers."
    #print(msg)

    # Step loss history
    if slack != 0:
        step_loss_history = SSPStepLossHistory(slack, n_workers)
    else:
        step_loss_history = SynchStepLossHistory(n_workers)

    # Storage backend
    storage = S3Backend(bucket_name='mlless2',
                    aws_access_key_id='************',
                    aws_secret_access_key='**************',
                    region_name='us-east-1')  # Specify the correct AWS region
                    
    # storage = RedisBackend(redis_hosts=['localhost'], redis_port=6379)
    # amqp_params = pika.ConnectionParameters(
    # host= 'localhost',
    # port=5672,
    # credentials=pika.PlainCredentials('guest', 'guest'))
    ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    amqp_params = pika.ConnectionParameters(host=broker_url,
                                       virtual_host='/',
                                       credentials=credentials,ssl_options=pika.SSLOptions(ssl_context))



    # RabbitMQ parameters
    communicator = SupervisorCommunicator(executor_id, slack, amqp_params)
    communicator.send_debug(msg)

    epochs = communicator.listen_workers(n_workers, step_loss_history, threshold, epochs, n_minibatches, remove_threshold,remove_interval, init_time, max_time)
    
    #print(f"supervisor received {epochs}")

    final_model, worker_times = communicator.listen_results(n_workers, storage)

    end_time = time.time()

    communicator.send_results(final_model, step_loss_history.loss_history, step_loss_history.step_time_history,
                              worker_times, end_time - init_time, epochs)

    communicator.channel.queue_delete(queue='{}_supervisor'.format(executor_id))
    communicator.channel.queue_delete(queue='{}_results'.format(executor_id))
    communicator.channel.queue_delete(queue='{}_client'.format(executor_id))

    print("[supervisor]: Finished supervisor function execution")


class StepLossHistory:
    def __init__(self, n_workers):
        self.loss_history = []
        self.step_time_history = []
        self.n_workers = n_workers
        self.step_time = time.time()
        self.step = 0

    def update_loss_history(self, step, loss, minibatch_size):
        pass

    def check_step_end(self, step):
        pass

    def get_loss(self, step):
        return self.loss_history[step]

    def get_time(self, step):
        return self.step_time_history[step]

    def set_update_available(self, w_id, step, update_available):
        pass

    def get_update_available(self, step):
        pass


class SynchStepLossHistory(StepLossHistory):
    def __init__(self, n_workers):
        self.count = 0
        self.loss = 0
        self.minibatch_size = 0
        self.worker_update_available = {}
        super().__init__(n_workers)

    def update_loss_history(self, _, loss, minibatch_size):
        self.loss += loss
        self.minibatch_size += minibatch_size
        self.count += 1

        return self.count

    def check_step_end(self, _):
        if self.count == self.n_workers:
            self.loss_history.append(self.loss / self.minibatch_size)
            self.step_time_history.append(time.time() - self.step_time)
            self.step_time = time.time()
            self.loss = 0
            self.minibatch_size = 0
            self.count = 0
            return True
        else:
            return False

    def set_update_available(self, w_id, _, update_available):
        if update_available:
            self.worker_update_available[w_id] = update_available

    def get_update_available(self, _):
        return self.worker_update_available

    def clear_update_available(self, _):
        self.worker_update_available = {}


class SSPStepLossHistory(StepLossHistory):
    def __init__(self, slack, n_workers):
        super().__init__(n_workers)
        self.slack_step = np.zeros(slack + 1, dtype=np.int)
        self.loss = np.zeros(slack + 1, dtype=np.int)
        self.minibatch_size = np.zeros(slack + 1, dtype=np.int)
        self.worker_update_available = [{}] * (slack + 1)
        self.slack = slack
        self.current_slack_step = 0
        self.processed_updates = np.zeros(slack + 1, dtype=np.int)

    def update_loss_history(self, step, loss, minibatch_size):
        mod_step = step % (self.slack + 1)
        self.loss[mod_step] += loss
        self.minibatch_size[mod_step] += minibatch_size
        self.slack_step[mod_step] += 1

        return self.slack_step[mod_step]

    def check_step_end(self, step):
        mod_step = step % (self.slack + 1)
        if self.slack_step[mod_step] == self.n_workers:
            self.loss_history.append(self.loss[mod_step] / self.minibatch_size[mod_step])
            self.step_time_history.append(time.time() - self.step_time)
            self.step_time = time.time()
            self.loss[mod_step] = 0
            self.minibatch_size[mod_step] = 0
            self.slack_step[mod_step] = 0
            return True
        else:
            return False

    def set_update_available(self, w_id, step, update_available):
        if update_available:
            self.worker_update_available[step % (self.slack + 1)][w_id] = update_available

    def get_update_available(self, step):
        return self.worker_update_available[step % (self.slack + 1)]

    def clear_update_available(self, step):
        self.worker_update_available[step % (self.slack + 1)] = {}

    def increment_processed_updates(self, step):
        self.processed_updates[step % (self.slack + 1)] += 1

    def zero_processed_updates(self, step):
        self.processed_updates[step % (self.slack + 1)] = 0

    def get_processed_updates(self, step):
        return self.processed_updates[step % (self.slack + 1)]
