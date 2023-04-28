import sys 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim # Optimizers

def dataset():
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

    batch_size = 256
    global train_loader,test_loader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(3*32*32, 1536)  # Input is image with shape (3*32*32)
        self.fc1_bn = nn.BatchNorm1d(1536)
        self.drop1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(1536, 768)  # First HL
        self.fc2_bn = nn.BatchNorm1d(768)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(768, 384) # second HL
        self.fc3_bn = nn.BatchNorm1d(384)
        self.drop3 = nn.Dropout(p=0.2)

        self.fc7 = nn.Linear(384, 10) # fourth HL

    def forward(self, x):
      # Batch x of shape (B, C, W, H) #batch size , colour channel and width and Height
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = self.fc1(x)
      x = F.relu(self.fc1_bn(x))  # First Hidden Layer
      x = self.drop1(x)
      x = self.fc2(x)
      x = F.relu(self.fc2_bn(x))  # Second Hidden Layer
      x = self.drop2(x)
      x = self.fc3(x)
      x = F.relu(self.fc3_bn(x))  # 3rd Hidden Layer
      x = self.drop3(x)
      x = self.fc7(x)  # Output Layer
      return x  # Has shape (B, 10)
    
def model_setup():
    # Identify device
    global device
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Creat the model and send its parameters to the appropriate device
    global mlp
    mlp = MLP().to(device)


# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device): # criterion loss function
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
            _, predicted = torch.max(outputs.data, 1)  # Get max value #index of predicition
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

def epochtraining():
    LEARNING_RATE = 1e-2
    MOMENTUM = 0.99

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM) # when you updating the weights
    lr_decay = optim.lr_scheduler.StepLR(optimizer, 10 , 0.1)

    # Train the MLP for 15 epochs
    for epoch in range(15):
        train_loss = train(mlp, train_loader, criterion, optimizer, device)
        test_acc = test(mlp, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

def savep():
    print("saving model...")
    torch.save(mlp.state_dict(), "MLPsavedmodel.pt")
    print("Done!")

def loadp():
    print("Loading params...")
    model = MLP()
    model.load_state_dict(torch.load("MLPsavedmodel.pt"))
    model.eval()
    print("Done!")
    #test_acc = test_acc*100
    #print(f"Test accuracy = {test_acc*100:.2f}%")

def main():
    dataset()
    model_setup()
    epochtraining()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mystring = sys.argv[1]
        save = "-save"
        load = "-load"
        if mystring == save:
            main()
            savep()
        else:
            loadp()
    else:
        main()