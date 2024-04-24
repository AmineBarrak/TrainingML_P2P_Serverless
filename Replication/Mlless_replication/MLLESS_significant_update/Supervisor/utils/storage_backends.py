import pickle
import zlib
import re 
import boto3
import pickle
import zlib

class StorageBackend:
    def __init__(self, compression=True, **kwargs):
        self.compression = compression

    def put(self, key, object_):
        data = pickle.dumps(object_)
        if self.compression:
            data = zlib.compress(data)
        self._put(key, data)

    def _put(self, key, object_):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")

    def get(self, key):
        data = self._get(key)
        if self.compression:
            data = zlib.decompress(data)
        data = pickle.loads(data)
        return data

    def _get(self, key):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")

    def delete(self, keys):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")


class RedisBackend(StorageBackend):
    def __init__(self, compression=True, n_redis=1, **kwargs):
        super().__init__(compression)



        redis_host = kwargs['redis_hosts'][0]
        port = kwargs.get('redis_port', 6379)  
        self.storage = Redis(host=redis_host, port=port)

        self.n_redis = n_redis
       

    def _put(self, key, object_):

        self.storage.set(key, object_)

    def _get(self, key):

        object_ = self.storage.get(key)
        return object_

    def delete(self, keys):

            self.storage.delete(*keys)

    def _extract_host(self, key):
        w_id = re.findall(r'_w(\d*)', key)[0]
        w_id = int(w_id)
        redis_host = w_id % self.n_redis
        return redis_host



class S3Backend(StorageBackend):
    def __init__(self, bucket_name, aws_access_key_id, aws_secret_access_key, compression=True, region_name='us-east-1', **kwargs):
        super().__init__(compression)
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
        self.bucket_name = bucket_name

    def _put(self, key, object_):
        self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=object_)

    def _get(self, key):
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        object_ = response['Body'].read()
        return object_

    def delete(self, keys):
        objects = [{'Key': key} for key in keys]
        self.s3.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects})
