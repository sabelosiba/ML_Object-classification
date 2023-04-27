import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim # Optimizers

# Create the transform sequence
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.5,), (0.5,)) 
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

batch_size = 16

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5 ) # First Conv Layer #input channel(colour channel) #output channel # kernel size # 5 trainable filters
        self.pool = nn.MaxPool2d(2, stride= 2)  # For pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # Second Conv Layer
        self.pool2 = nn.MaxPool2d(2, stride= 2)  # For pooling
        self.flatten = nn.Flatten() # For flattening the 2D image

        self.fc1 = nn.Linear(400, 120) # input is image with 16*5*5 from conv2
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)  # First FC HL
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3= nn.Linear(84, 10) # Output layer

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 28, 28)s
      x = self.pool(x)  # Shape: (B, 5, 14, 14)
      x = F.relu(self.conv2(x))
      x = self.pool2(x)
      x = self.flatten(x) # Shape: (B, 980)

      x = self.fc1(x)
      x = F.relu(self.fc1_bn(x))  # Shape (B, 256)
      x = self.fc2(x)
      x = F.relu(self.fc2_bn(x))  # Shape (B, 256)
      x = self.fc3(x)  # Shape: (B, 10)
      return x  

# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

cnn = CNN().to(device)


# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

    
cnn = CNN().to(device)

LEARNING_RATE = 1e-1
MOMENTUM = 0.6

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer, 5 , 0.1)
# Train the MLP for 5 epochs
for epoch in range(10):
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    lr_decay.step()
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")