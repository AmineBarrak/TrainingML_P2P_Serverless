import json

def create_parallel_branch(index):
    return {
        "StartAt": f"Batch{index}",
        "States": {
            f"Batch{index}": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:us-east-1:736943246631:function:compute_gradients",
                "Parameters": {
                    "rank.$": "$$.Execution.Input.rank",
                    "size.$": "$$.Execution.Input.size",
                    "batch_rank.$": f"$$.Execution.Input.batch_rank[{index - 1}]",
                    "ec2_ip.$": "$$.Execution.Input.ec2_ip",
                    "port.$": "$$.Execution.Input.port",
                    "dataset.$": "$$.Execution.Input.dataset",
                    "model_str.$": "$$.Execution.Input.model_str",
                    "optimiser.$": "$$.Execution.Input.optimiser",
                    "optimiser_lr.$": "$$.Execution.Input.optimiser_lr",
                    "loss.$": "$$.Execution.Input.loss",
                    "username.$": "$$.Execution.Input.username",
                    "password.$": "$$.Execution.Input.password",
                    "path_key.$": "$$.Execution.Input.path_key",
                    "num_batches.$": "$$.Execution.Input.num_batches",
                    "num_peers.$": "$$.Execution.Input.num_peers",
                    "epoch.$": "$$.Execution.Input.epoch",
                    "attack.$": "$$.Execution.Input.attack"
                },
                "End": True
            }
        }
    }

def create_step_function(num_batches):
    state_machine = {
        "Comment": "Lambda execution workflow",
        "StartAt": "Authentification",
        "States": {
        "Authentification": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:us-east-1:736943246631:function:auth",
                "Parameters": {
                    "waiting_queue.$": "$$.Execution.Input.waiting_queue",
                    "password_queue.$": "$$.Execution.Input.password_queue",
                    "username.$": "$$.Execution.Input.username",
                    "ec2_ip.$": "$$.Execution.Input.ec2_ip",
                    "port.$": "$$.Execution.Input.port",
                    "password.$": "$$.Execution.Input.password",
                    "path_key.$": "$$.Execution.Input.path_key"
                },
        "Next": "ComputeGradients"
            },
            "ComputeGradients": {
                "Type": "Parallel",
                "Branches": [create_parallel_branch(i) for i in range(1, num_batches + 1)],
                "Next": "AverageGradients"
            },

         "AverageGradients": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:736943246631:function:Trigger_Average_Gradient_sync",
      "Parameters": {
        "rank.$": "$$.Execution.Input.rank",
        "ec2_ip.$": "$$.Execution.Input.ec2_ip",
        "port.$": "$$.Execution.Input.port",
        "username.$": "$$.Execution.Input.username",
        "password.$": "$$.Execution.Input.password",
        "path_key.$": "$$.Execution.Input.path_key",
        "num_batches.$": "$$.Execution.Input.num_batches",
        "num_peers.$": "$$.Execution.Input.num_peers",
        "epoch.$": "$$.Execution.Input.epoch",
        "attack.$": "$$.Execution.Input.attack"
      },

       "Next": "CheckHeartbeat"
    },
    "CheckHeartbeat": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:736943246631:function:check_heartbeat",
      "Parameters": {
        "username.$": "$$.Execution.Input.username",
        "path_key.$": "$$.Execution.Input.path_key",
        "ec2_ip.$": "$$.Execution.Input.ec2_ip",
        "port.$": "$$.Execution.Input.port",
        "password.$": "$$.Execution.Input.password",
        "epoch.$": "$$.Execution.Input.epoch"
      },
      "Next": "Aggregation"
    },
    "Aggregation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:736943246631:function:aggregation",
      "Parameters": {
        "ec2_ip.$": "$$.Execution.Input.ec2_ip",
        "password.$": "$$.Execution.Input.password",
        "username.$": "$$.Execution.Input.username",
        "path_key.$": "$$.Execution.Input.path_key",
        "port.$": "$$.Execution.Input.port",
        "epoch.$": "$$.Execution.Input.epoch"
      },
      "Next": "Update"
    },
    "Update": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:736943246631:function:Trigger_update_model",
      "Parameters": {
        "grad_key.$": "$$.Execution.Input.grad_key",
        "param_key.$": "$$.Execution.Input.param_key",
        "ec2_ip.$": "$$.Execution.Input.ec2_ip",
        "port.$": "$$.Execution.Input.port",
        "password.$": "$$.Execution.Input.password",
        "username.$": "$$.Execution.Input.username",
        "path_key.$": "$$.Execution.Input.path_key",
        "epoch.$": "$$.Execution.Input.epoch"
      },
      "Next": "CheckConvergence"
    },
    "CheckConvergence": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:736943246631:function:check-convergence",
      "Parameters": {
        "waiting_queue.$": "$$.Execution.Input.waiting_queue",
        "password_queue.$": "$$.Execution.Input.password_queue",
        "username.$": "$$.Execution.Input.username",
        "ec2_ip.$": "$$.Execution.Input.ec2_ip",
        "port.$": "$$.Execution.Input.port",
        "password.$": "$$.Execution.Input.password",
        "path_key.$": "$$.Execution.Input.path_key",
        "rank.$": "$$.Execution.Input.rank",
        "size.$": "$$.Execution.Input.size",
        "batch_rank.$": "$$.Execution.Input.batch_rank",
        "dataset.$": "$$.Execution.Input.dataset",
        "model_str.$": "$$.Execution.Input.model_str",
        "optimiser.$": "$$.Execution.Input.optimiser",
        "optimiser_lr.$": "$$.Execution.Input.optimiser_lr",
        "loss.$": "$$.Execution.Input.loss",
        "num_batches.$": "$$.Execution.Input.num_batches",
        "num_peers.$": "$$.Execution.Input.num_peers",
        "grad_key.$": "$$.Execution.Input.grad_key",
        "param_key.$": "$$.Execution.Input.param_key",
        "bucket.$": "$$.Execution.Input.bucket",
        "batch_size.$": "$$.Execution.Input.batch_size",
        "epoch.$": "$$.Execution.Input.epoch",
        "attack.$": "$$.Execution.Input.attack"
      },
      "Next": "trigger_next"
      
      
    },
        "trigger_next": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:us-east-1:736943246631:function:Update_Trigger_Next_epoch",
          "Parameters": {
            "grad_key.$": "$$.Execution.Input.grad_key",
            "param_key.$": "$$.Execution.Input.param_key",
            "ec2_ip.$": "$$.Execution.Input.ec2_ip",
            "port.$": "$$.Execution.Input.port",
            "password.$": "$$.Execution.Input.password",
            "username.$": "$$.Execution.Input.username",
            "path_key.$": "$$.Execution.Input.path_key",
            "epoch.$": "$$.Execution.Input.epoch"
          },
        "End": True
        }
        }
    }
    return state_machine


if __name__ == "__main__":
    num_batches = 20  # Define the number of batches/gradients to compute in parallel
    sf_definition = create_step_function(num_batches)
    with open("state_machine_definition.json", 'w') as file:
    	json.dump(sf_definition, file, indent=4)
