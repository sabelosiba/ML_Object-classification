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
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResidualBlock(nn.Module):
     def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.linear = nn.Linear(512* block.expansion, 10)

     def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

     def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def model_setup():
    # Identify device
    global device, resnet
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    resnet = ResidualBlock(BasicBlock).to(device)

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


def epoch_resnet():
    LEARNING_RATE = 1e-1
    MOMENTUM = 0.9

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
    optimizer = optim.SGD(resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_decay = optim.lr_scheduler.StepLR(optimizer, 5 , 0.1)
    # Train the MLP for 10 epochs
    for epoch in range(10):
        train_loss = train(resnet, train_loader, criterion, optimizer, device)
        test_acc = test(resnet, test_loader, device)
        lr_decay.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

def savep():
    print("saving model...")
    torch.save(resnet.state_dict(), "RESNETsavedmodel.pt")
    print("Done!")

def loadp():
    print("Loading params...")
    model = ResidualBlock()
    model.load_state_dict(torch.load("RESNETsavedmodel.pt"))
    model.eval()
    print("Done!")
    #test_acc = test_acc*100
    #print(f"Test accuracy = {test_acc*100:.4f}%")

def main():
    dataset()
    model_setup()
    #epochtraining()
    epoch_resnet()

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