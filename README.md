# Comparative Overview of Serverless Machine Learning Frameworks

## Introduction to Serverless Computing
Serverless computing has revolutionized the way distributed machine learning (ML) tasks are performed by eliminating the need for managing servers and allowing developers to focus purely on code. This document explores various serverless frameworks that have been specifically designed to enhance distributed ML tasks in a serverless environment.

## Frameworks Description

### SPIRT
![SPIRT Architecture](./SPIRT/Spirt.png "SPIRT Framework Architecture")

SPIRT operates on a peer-based system where each worker maintains its own database and a serverless workflow orchestrated by AWS Step Function. The workflow encompasses:
1. Fetching minibatches.
2. Parallel gradient computation.
3. Storing gradients in the worker's database.
4. Averaging gradients within the database.
5. Notifying completion via a synchronization queue.
6. Polling the synchronization queue.
7. Retrieving averaged gradients from peer databases.
8. Aggregating these averages.
9. Updating local models.

### MLLESS
![MLLESS Architecture](./Replication/Mlless_replication/MLLESS.png "MLLESS Framework Architecture")

In MLLESS, the workflow includes:
1. Fetching a minibatch.
2. Computing the gradient.
3. Storing significant gradients in a shared database.
4. Monitoring queues for updates.
5. Waiting for all updates as communicated by the supervisor.
6. Fetching and aggregating the corresponding gradients.
7. Updating the model with the aggregated gradients.

### ScatterReduce-LambdaML
![ScatterReduce-LambdaML Architecture](./Replication/ScatterReduce_replication/LambdaScatter_Reduce.png "ScatterReduce-LambdaML Framework Architecture")

This approach involves:
1. Fetching minibatches.
2. Computing gradients.
3. Dividing and distributing gradient chunks.
4. Fetching and aggregating assigned chunks.
5. Sending aggregated chunks back to the database.
6. Retrieving and concatenating all aggregated chunks.
7. Updating the model with the complete gradient.

### AllReduce-LambdaML
![AllReduce-LambdaML Architecture](./Replication/AllReduce_replication/LambdaAll_reduce.png "AllReduce-LambdaML Framework Architecture")

The AllReduce-LambdaML framework proceeds as follows:
1. Fetching a minibatch.
2. Computing gradients.
3. Sending gradients to a shared database.
4. Aggregating all gradients into a single unified gradient (performed by a designated master worker).
5. Sending the aggregated gradient back to the database.
6. Each worker updates their local models with the aggregated gradient.


## Execution of the Frameworks:
### SPIRT
### MLLESS
### ScatterReduce-LambdaML
### AllReduce-LambdaML

## Publications:
This work was published in the following:

<div style="border: 1px solid grey; padding: 10px;">

**Citation**

```bibtex
@INPROCEEDINGS{10366723,
  author={Barrak, Amine and Jaziri, Mayssa and Trabelsi, Ranim and Jaafar, Fehmi and Petrillo, Fabio},
  booktitle={2023 IEEE 23rd International Conference on Software Quality, Reliability, and Security (QRS)},
  title={SPIRT: A Fault-Tolerant and Reliable Peer-to-Peer Serverless ML Training Architecture},
  year={2023},
  keywords={Training; Fault tolerance; Scalability; Fault tolerant systems; Computer architecture; Machine learning; Robustness; Distributed Machine Learning; Peer-to-Peer (P2P); Serverless Computing; Fault Tolerance; Robust Aggregation},
  doi={10.1109/QRS60937.2023.00069}
}
</div>
<div style="border: 1px solid grey; padding: 10px; margin-top: 10px;">

**Under Review at**

IEEE Transactions on Parallel and Distributed Systems

</div>
```

  
  Under review at: IEEE Transactions on Parallel and Distributed Systems

  
  
  


