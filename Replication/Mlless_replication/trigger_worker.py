import boto3
import json
import argparse

def invoke_lambda(function_name, payload):
    # Initialize a boto3 client for AWS Lambda
    lambda_client = boto3.client('lambda')

    try:
        # Invoke the Lambda function with the given payload synchronously
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='Event',  # Synchronous invocation
            Payload=json.dumps(payload)
        )
        
        # Read the Lambda function response payload
        response_payload = json.load(response['Payload'])
        print(f"Response from {function_name}: {response_payload}")

    except Exception as e:
        print(f"Error invoking {function_name}: {e}")

def main(rank):
    n_workers = 4
    broker_url = "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"

    # Payload for the mlless-worker Lambda function
    worker_payload = {
        "worker_id": rank,
        "n_workers": n_workers,
        "broker_url": broker_url
    }

    # Invoke the mlless-worker Lambda function
    invoke_lambda('mlless-worker', worker_payload)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Invoke mlless-worker Lambda function with a given worker rank.')
    parser.add_argument('rank', type=int, help='The rank (or ID) of the worker')
    args = parser.parse_args()

    # Call the main function with the provided rank
    main(args.rank)

