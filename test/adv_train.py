# This is the test code for adversarial training with LeNet, MNIST Dataset, FGSM, PGD Adversarial Attack

# Environment setting, please following the below environment setting.
# conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
# pip install ftfy regex tqdm
# pip install opencv-python boto3 requests pandas

# Import all necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os


# Define a simple LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define two different adversarial attack methodologies
# We then define a fgsm_attack
def fgsm_attack(model, loss_fn, images, labels, epsilon):
    images = images.clone().detach().to(device).requires_grad_(True)
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

# Next, we define the PGD Attack
def pgd_attack(model, loss_fn, images, labels, epsilon, alpha, num_iter):
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True
    
    for _ in range(num_iter):
        outputs = model(perturbed_images)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * data_grad.sign()
        perturbation = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = torch.clamp(images + perturbation, 0, 1).detach_()
        perturbed_images.requires_grad = True
    
    return perturbed_images


print("Start Data Transformation and dataset preparing")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# If you encounter any issue when downloading the MNIST Dataset, please follow the post:
# https://github.com/pytorch/vision/issues/1938

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define the training function
def train(model, train_loader, optimizer, loss_fn, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Define the testing function
def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


print("Start Adversarial training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()
if os.path.exists("lenet_normal.pth"):
    print("Loading normal checkpoint")
else:
    # Initialize model, optimizer, and loss function
    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Train the model normally
    for epoch in range(1, 6):
        train(model, train_loader, optimizer, loss_fn, epoch, device)
        test(model, test_loader, loss_fn, device)

    # Save the normal model
    torch.save(model.state_dict(), "lenet_normal.pth")

# Adversarial training
if os.path.exists("lenet_adversarial.pth"):
    print("loading model with adversarial training checkpoint")
else:
    for epoch in range(1, 6):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Apply PGD attack
            perturbed_data = pgd_attack(model, loss_fn, data, target, epsilon=0.3, alpha=0.01, num_iter=40)
            
            # Train on adversarial examples
            optimizer.zero_grad()
            output = model(perturbed_data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Adversarial Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

        test(model, test_loader, loss_fn, device)

    # Save the adversarially trained model
    torch.save(model.state_dict(), "lenet_adversarial.pth")


print("Starting testing after the adversarial attack: ")
# Load the normal model
model_normal = LeNet().to(device)
model_normal.load_state_dict(torch.load("lenet_normal.pth"))

# Load the adversarially trained model
model_adv = LeNet().to(device)
model_adv.load_state_dict(torch.load("lenet_adversarial.pth"))

def test_attack(model, test_loader, loss_fn, attack_fn, attack_params, device):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        perturbed_data = attack_fn(model, loss_fn, data, target, **attack_params)
        output = model(perturbed_data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    accuracy = correct / total
    print(f'Accuracy under attack: {accuracy * 100:.2f}%')

# FGSM attack parameters
fgsm_params = {'epsilon': 0.3}

# PGD attack parameters
pgd_params = {'epsilon': 0.3, 'alpha': 0.01, 'num_iter': 40}

print("Normal model:")
print("Under FGSM Attack")
test_attack(model_normal, test_loader, loss_fn, fgsm_attack, fgsm_params, device)
print("Under PGD Attack")
test_attack(model_normal, test_loader, loss_fn, pgd_attack, pgd_params, device)

print("Adversarially trained model:")
print("Under FGSM Attack")
test_attack(model_adv, test_loader, loss_fn, fgsm_attack, fgsm_params, device)
print("Under PGD Attack")
test_attack(model_adv, test_loader, loss_fn, pgd_attack, pgd_params, device)