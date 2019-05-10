import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() # new added
        self.pool = nn.MaxPool2d(kernel_size=8, padding=0)
        #self.relu2 = nn.ReLU() # new added



        self.fc1 = nn.Linear(32 * 56 * 75, 128)
        self.relu3 = nn.ReLU() # new added
        self.fc2 = nn.Linear(128, num_classes)
        self.sf1 = nn.Softmax() # new added

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 56 * 75)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)