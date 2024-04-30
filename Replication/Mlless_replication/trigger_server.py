import boto3
import json

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

def main():
    n_workers = 4
    broker_url = "b-0b3e161b-54da-47bc-b2ef-05692144f6a9.mq.us-east-1.amazonaws.com"

    # Payload for the mlless_server Lambda function
    server_payload = {
        "n_workers": n_workers,
        "broker_url": broker_url
    }

    # Invoke the mlless_server Lambda function
    invoke_lambda('mlless_server', server_payload)

if __name__ == "__main__":
    main()

