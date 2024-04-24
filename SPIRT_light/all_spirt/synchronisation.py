import pickle
import pika
import ssl
import threading
import time

def receive_n_messages_with_timeout(channel, queue_name, n, timeout=500):
    # Ensure the queue exists
    channel.queue_declare(queue=queue_name, durable=True)

    received_messages = []
    print(f'Waiting to receive {n} messages from {queue_name} or timeout after {timeout} seconds.')

    start_time = time.time()
    while len(received_messages) < n:
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)

        if method_frame:
            message = body.decode('utf-8')
            print(f"Received: {message}")
            received_messages.append(message)
        else:
            # Check for timeout
            current_time = time.time()
            if (current_time - start_time) > timeout:
                print("Timeout reached. Exiting.")
                break
            else:
                print("No message in the queue. Waiting...")
                time.sleep(1)  # Wait for 1 second before checking again

    print("Finished receiving messages.")
    return received_messages


def wait_for_n_messages_or_timeout(channel, queue_name, n, timeout=500):
    start_time = time.time()

    print(f"Waiting for at least {n} messages in '{queue_name}' or timeout after {timeout} seconds.")

    while True:
        # Check the queue without consuming messages
        queue = channel.queue_declare(queue=queue_name, passive=True)
        message_count = queue.method.message_count

        if message_count >= n:
            print(f"Found {message_count} messages in the queue. Proceeding...")
            return True  # Return True if n messages are found

        # Check for timeout
        if (time.time() - start_time) > timeout:
            print("Timeout reached. Exiting.")
            return False  # Return False if timeout is reached

        # Wait a bit before checking again to avoid hammering the server
        time.sleep(1)


def empty_queue(channel, queue_name):
    # Purge all messages from the specified queue
    method_frame = channel.queue_purge(queue=queue_name)

    # method_frame.method.message_count contains the number of messages purged
    purged_message_count = method_frame.method.message_count

    print(f"Queue '{queue_name}' emptied. Total messages purged: {purged_message_count}")
    return purged_message_count

def send_message(channel, queue_name, message):
    # Ensure the queue exists
    #channel.queue_declare(queue=queue_name, durable=True)
    #sleep(1)

    # Convert the message to bytes
    message_bytes = message.encode('utf-8')

    # Publish the message to the queue
    channel.basic_publish(exchange='',
                          routing_key=queue_name,
                          body=message_bytes,
                          properties=pika.BasicProperties(
                             delivery_mode=2,  # make message persistent
                          ))
    print(f"Sent: {message}")