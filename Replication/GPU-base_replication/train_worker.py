# Mount Google Drive once at the top
from google.colab import drive
drive.mount('/content/drive')

import os
import time
import io
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import boto3

# ---------- Configuration ----------
COLLAB_LOG_DIR = "/content/drive/MyDrive/collab"
os.makedirs(COLLAB_LOG_DIR, exist_ok=True)
log_path = os.path.join(COLLAB_LOG_DIR, "minibatch_log.csv")
if os.path.exists(log_path):
    os.remove(log_path)

S3_BUCKET = "YOUR_S3_BUCKET_NAME"  
S3_PREFIX = "gradients"             # folder in bucket

# Initialize S3 client
s3 = boto3.client('s3')

# ---------- Helper functions ----------
def upload_gradients(grad_list, minibatch, worker_id):
    """Serialize and upload a list of tensors to S3."""
    key = f"{S3_PREFIX}/minibatch_{minibatch}_worker_{worker_id}.pt"
    buf = io.BytesIO()
    torch.save(grad_list, buf)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())

def download_gradients(minibatch, worker_id):
    """Download and deserialize a list of tensors from S3."""
    key = f"{S3_PREFIX}/minibatch_{minibatch}_worker_{worker_id}.pt"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    buf = io.BytesIO(resp['Body'].read())
    return torch.load(buf)

# ---------- MobileNet definition (unchanged) ----------
class MobileNet_Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                               padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MobileNet(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2),
           512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes, stride = (x, 1) if isinstance(x, int) else (x[0], x[1])
            layers.append(MobileNet_Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- Data ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2
)

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ---------- Training ----------
num_workers = 4
epochs = 15
minibatch_counter = 0
logs = []
correct, total = 0, 0

for epoch in range(1, epochs+1):
    print(f"\n--- Epoch {epoch}/{epochs} ---")
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # split batch equally
        inputs_split = torch.chunk(inputs, num_workers)
        labels_split = torch.chunk(labels, num_workers)

        # 1) Each worker computes its gradients and uploads them
        for w in range(num_workers):
            worker_model = copy.deepcopy(model).to(device)
            optimizer.zero_grad()

            start = time.time()
            outputs = worker_model(inputs_split[w])
            loss = criterion(outputs, labels_split[w])
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            worker_time = time.time() - start

            # collect gradients
            grad_list = [p.grad.data.clone().cpu() for p in worker_model.parameters() if p.requires_grad]
            upload_gradients(grad_list, minibatch_counter, w)

            # log time
            logs.append({
                "minibatch": minibatch_counter,
                "worker_id": w,
                "worker_time": worker_time,
                "accuracy_so_far": None
            })

        # 2) Download all 4 sets, average, and apply update
        all_grads = []
        for w in range(num_workers):
            all_grads.append(download_gradients(minibatch_counter, w))

        # average per-parameter
        avg_grads = []
        for idx in range(len(all_grads[0])):
            stacked = torch.stack([g[idx] for g in all_grads], dim=0)
            avg = stacked.mean(dim=0).to(device)
            avg_grads.append(avg)

        # assign to central model
        optimizer.zero_grad()
        for p, g in zip(model.parameters(), avg_grads):
            p.grad = g
        optimizer.step()

        # 3) compute running accuracy
        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        acc = 100 * correct / total

        # update logs accuracy
        for i in range(num_workers):
            logs[-1 - i]["accuracy_so_far"] = acc

        # print & save
        worker_times = [round(logs[-num_workers + i]["worker_time"], 3) for i in range(num_workers)]
        print(f"Batch {minibatch_counter:04d} | Acc: {acc:.2f}% | Times: {worker_times} s")
        pd.DataFrame(logs).to_csv(log_path, index=False)

        minibatch_counter += 1

print(f"\n✅ Final log saved at: {log_path}")
