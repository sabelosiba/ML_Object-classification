import sys 
import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim # Optimizers

def dataset():
    # Create the transform sequence
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        # Normalize Image to [-1, 1] first number is mean, second is std deviation
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # Load CIFAR10 dataset
    # Train
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

    batch_size = 128
    global train_loader,test_loader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Define the RESNET architecture
"""
Define an nn.Module class for a simple residual block with equal dimensions
"""
class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(64)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
   
    """
    Combine output with the original input
    """
    def forward(self, x): return x + self.convblock(x) # skip connection

def model_setup():
    # Identify device
    global device, resnet
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #resnet = ResidualBlock(BasicBlock).to(device)
    resnet = torchvision.models.resnet34(pretrained=True)

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


def epoch_resnet(flag):
    LEARNING_RATE = 1e-1
    MOMENTUM = 0.9

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
    optimizer = optim.SGD(resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_decay = optim.lr_scheduler.StepLR(optimizer, 5 , 0.1)
    global max 
    max=0
    # Train the MLP for 10 epochs
    for epoch in range(10):
        train_loss = train(resnet, train_loader, criterion, optimizer, device)
        test_acc = test(resnet, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
        if test_acc > max:
            max = test_acc
    if flag == True:
            savep(max)

def savep(maxx):
    print("saving model...")
    mymax = maxx*100
    torch.save({'Testing accuracy': mymax,
                'model_state_dict': resnet.state_dict(),
                }, "RESNETsavedmodel.pt")
    print("Done!")

def loadp():
    print("Loading params...")
    model = ResBlock()
    check = torch.load("RESNETsavedmodel.pt")
    model.load_state_dict(check['model_state_dict'])
    mymax = check['Testing accuracy']
    print("Done!")
    print(f"Test accuracy = {mymax:.2f}%")

def main(b):
    dataset()
    model_setup()
    #epochtraining()
    epoch_resnet(b)

if __name__ == "__main__":
    b = False
    if len(sys.argv) > 1:
        mystring = sys.argv[1]
        save = "-save"
        load = "-load"
        if mystring == save:
            b = True
            main(b)
        else:
            loadp()
    else:
        main(b)