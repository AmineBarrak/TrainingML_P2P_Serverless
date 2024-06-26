# -*- coding: utf-8 -*-
"""ModelManager.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HpkkXKOiNmvm-ZbQ8-FOxR6STSkR9QIp
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets 
from torch.distributed.rpc import RRef, remote, rpc_async 
from time import sleep, time
import sys


class MobileNet_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNet_Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def select_loss(loss_fn):

  losses = {"NLL":nn.NLLLoss,"cross_entropy":nn.CrossEntropyLoss} 
  if loss_fn in losses.keys(): 
    return losses[loss_fn]() 
  else: 
    print("The selected loss function is undefined, available losses are: ", losses.keys())

def select_model(dataset,model):
  models={ 'resnet18':torchvision.models.resnet18,
                'resnet34':torchvision.models.resnet34,
                'resnet50':torchvision.models.resnet50,
                'resnet152':torchvision.models.resnet152,
                            'inception':torchvision.models.inception_v3,
                             'vgg16':torchvision.models.vgg16,
                             'vgg19':torchvision.models.vgg19, 
                            'vgg11':torchvision.models.vgg11,
                            'squeeznet11':torchvision.models.squeezenet1_1,
                             'mobilenet-v2':torchvision.models.mobilenet_v2,
                             'mnasnet0_5':torchvision.models.mnasnet0_5,
                             'densenet121': torchvision.models.densenet121, 
                             'mobilenet-v3-small': torchvision.models.mobilenet_v3_small,
                             'mobilenet' : MobileNet
                  
                            
      
  } 
  datasets= { "cifar10":10,"mnist":10} 
  if dataset in datasets.keys():
    num_classes = datasets[dataset] 
  else:
    print("The specified dataset is undefined, available datasets are: ", datasets.keys())


  if model in models.keys():
    model = models[model](num_classes=num_classes) 
  else:
    print("The specified model is undefined, available models are: ", models.keys())


  return model

def select_optimizer(model,optimizer,lr):
  optimizers={'sgd': optim.SGD,
                'adam': optim.Adam,
                'adamw':optim.AdamW,
                'rmsprop': optim.RMSprop,
                'adagrad': optim.Adagrad} 
  if optimizer in optimizers.keys():
    return optimizers[optimizer](model.parameters(),lr=lr) 
    
    
    
    
    
class ModelPytorch:
    def __init__(self, dataset, model_str):
        self.model = select_model(dataset,model_str)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        #self.accumulated_gradients = {name: torch.zeros_like(param) for name, param in self.named_parameters()}
        self.accumulated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        self.mask_significance = {name: False for name, _ in self.model.named_parameters()}
        self.total_variation_accumulated = 0.0  # Initialize accumulated variation


    def step(self, epoch, step, minibatch):


        features = minibatch['images']

        labels = minibatch['labels']
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)


        loss.backward() 
        
        # Collect gradients in their original shape
        gradients = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]

         # Check if gradients are not empty and print their sum as a simple verification
        if gradients:
            print(f"Gradients are not empty. Sample gradient sum: {gradients[0].sum()}")
        else:
            print("Gradients are empty!")

        # Optionally, you can also print the shape of the first gradient to understand its dimensions
        if gradients:
            print(f"Shape of the first gradient: {gradients[0].shape}")

        # Accumulate gradients
        with torch.no_grad():  # Ensure no computation graph is created for this operation
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.accumulated_gradients:
                        self.accumulated_gradients[name] = param.grad.clone()
                    else:
                        self.accumulated_gradients[name] += param.grad

        # Print a sample from the accumulated gradients to verify accumulation
        sample_param_name = next(iter(self.accumulated_gradients))
        print(f"Accumulated gradient sum for '{sample_param_name}': {self.accumulated_gradients[sample_param_name].sum()}")

        

        
        return loss, gradients, outputs, labels
        
        
    

           
                    
                    
    def get_significant_gradients(self, threshold=1.0):
        # Initialize lists to hold all gradients and parameters
        all_gradients = []
        all_parameters = []

        # Iterate over all parameters and accumulate their gradients and values
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Accumulate gradients for the norm calculation
                all_gradients.append(self.accumulated_gradients[name].flatten())

                # Accumulate parameters for the norm calculation
                all_parameters.append(param.data.flatten())

        # Concatenate all gradients and parameters into flat tensors
        all_gradients_concat = torch.cat(all_gradients)
        all_parameters_concat = torch.cat(all_parameters)

        # Calculate the norms of the concatenated tensors
        accumulated_grad_norm = torch.norm(all_gradients_concat).item() + 1e-8
        param_norm = torch.norm(all_parameters_concat).item() + 1e-8

        # Update the total accumulated variation
        self.total_variation_accumulated += accumulated_grad_norm / param_norm
        print(f"Total accumulated variation: {self.total_variation_accumulated}")

        if self.total_variation_accumulated > threshold:
            # If the accumulated variation exceeds the threshold, process significant gradients
            significant_gradients = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Clone, detach, and flatten the accumulated gradient
                    significant_grad = self.accumulated_gradients[name].clone().detach().flatten()

                    # Reset the significant part of the accumulated gradients to zero
                    self.accumulated_gradients[name].zero_()

                    significant_gradients.append(significant_grad)

            # Reset the total accumulated variation
            self.total_variation_accumulated = 0.0

            # Concatenate all significant gradients into a single flat tensor
            significant_updates_concat = torch.cat(significant_gradients)
            return significant_updates_concat
        else:
            # If the accumulated variation does not exceed the threshold, return None
            print("Accumulated variation has not reached the threshold yet. Returning None.")
            return None
    
              


    def get_significant_updates(self, updates):
        # For simplicity, return all model parameters as updates
        return updates
        
    def apply_update(self, average):
            if average is not None:
                # Your existing logic when average is not None
                cur_pos = 0
                average_tensor = torch.cat([torch.flatten(grad) for grad in average])
                for param in self.model.parameters():
                    param_size = param.numel()
                    param.grad = average_tensor[cur_pos:cur_pos + param_size].view(param.size()).detach()
                    cur_pos += param_size
            else:
                # If average is None, do not modify gradients and just step
                pass

            # Perform the optimizer step outside the conditional
            # This ensures the optimizer step is done whether average is None or not
            self.optimizer.step()
    


    def get_weights(self):
        # Return the current weights of the model
        # You may need to adjust this based on how your model stores weights
        return [param.data for param in self.parameters()]

    def aggregate_model(self, received_weights):
        # Aggregate the received weights into the current model
        # You may need to adjust this based on how your model updates weights
        for param, received_param in zip(self.parameters(), received_weights):
            param.data += received_param

    def parameters(self):
        return [param for param in self.model.parameters() ] 



def get_model(dataset, model_str, **kwargs):
    return ModelPytorch(dataset, model_str)
