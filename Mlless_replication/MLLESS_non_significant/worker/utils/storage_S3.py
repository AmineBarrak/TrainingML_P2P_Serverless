import pickle
import time
import zlib
import random
import boto3
import inspect


class StorageIterator:
    def __init__(self, bucket, dataset, worker_id, n_minibatches, seed, n_workers):
        self.i = worker_id
        self.worker_id = worker_id
        self.s3 = boto3.client('s3', aws_access_key_id='*************', aws_secret_access_key='*************')
        self.rand = random.Random(seed)  # Used for pseudo-random minibatch shuffling
        self.minibatches = list(range(n_minibatches))
        self.dataset = dataset
        self.n_workers = n_workers
        self.n_minibatches = n_minibatches
        self.bucket = bucket

    def __iter__(self):
        return self

    def __next__(self): 
        # Inspect the call stack
        stack = inspect.stack()
        # The caller information is in the previous frame
        caller_frame = stack[1]
        caller_function = caller_frame.function
        caller_filename = caller_frame.filename
        caller_lineno = caller_frame.lineno
        print(f"Called by {caller_function} in {caller_filename}:{caller_lineno}")
        t0 = time.time()

        minibatch_id = self.minibatches[self.i]
        #print(f"self.i: {self.i}")

        object_key = f"{self.dataset}-part{minibatch_id}.pickle"
        print(f"self.i: {self.i}object is {object_key}")
        response = self.s3.get_object(Bucket=self.bucket, Key=object_key)
        minibatch = pickle.loads(zlib.decompress(response['Body'].read()))
        self.prev_i = self.i
        self.i = (self.i + self.n_workers) % self.n_minibatches

        # Epoch is over
        if self.i < self.prev_i:
            self.rand.shuffle(self.minibatches)
            print("end epoch")  
        t1 = time.time()
        self.cos_time = t1 - t0
        return minibatch

      
