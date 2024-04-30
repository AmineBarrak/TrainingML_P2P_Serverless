import boto3
import json

# Initialize a boto3 client for Lambda
lambda_client = boto3.client('lambda')

def trigger_lambdas(num_workers, data_bucket):
    for rank in range(num_workers):
        # Prepare the event data to pass to the Lambda function
        event_data = {
            "data_bucket": data_bucket,
            "rank": rank,
            "num_workers": num_workers
        }
        
        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName='scatterreduce',
            InvocationType='Event',  # Use 'RequestResponse' if you want to wait for the function to finish
            Payload=json.dumps(event_data).encode('utf-8')
        )
        
        # Print the response status code for debugging
        print(f"Invoked Lambda for worker {rank} with response status code: {response['StatusCode']}")

# Specify the number of workers and the data bucket name
num_workers = 16  # Example: 5 workers
data_bucket = "cld202-sagemaker"

# Trigger the Lambda functions
trigger_lambdas(num_workers, data_bucket)

