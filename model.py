
import torch.nn as nn
import torch.nn.functional as F
class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1,padding=2) # in: 32x32x1, out: 28x28x6
        # Subsampling (max pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # in: 14x14x6 , out: 10x10x16
        

        # Fully connected layers
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.flatten = nn.Flatten(start_dim=1)  # in: 400x120, out:1x400*120
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)  # 10 output classes (digits 0-9)
        self.activation = nn.ReLU()
        
    def forward(self, img):
        x=img
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
class LeNet5_normalize(nn.Module):
    """ LeNet-5 (LeCun et al., 1998) with Batch Normalization and Dropout for improved regularization.

    - For a detailed architecture, refer to the lecture note
    - Freely choose activation functions as you want
    - For subsampling, use max pooling with kernel_size = (2,2)
    - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) # Maintaining input size
        self.bn1 = nn.BatchNorm2d(6)  # Batch normalization after Conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after Conv2

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(120)  # Batch normalization after Conv3

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(0.)  # Dropout layer with 50% probability
        self.fc2 = nn.Linear(84, 10)  # Output layer for 10 classes

        self.activation = nn.ReLU()
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout1(x)  # Applying dropout before activation
        x = self.activation(x)

        x = self.fc2(x)
        return x
    
class CustomMLP(nn.Module):
    """ Your custom MLP model """
    
    def __init__(self):
        super(CustomMLP, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(784, 100)  # Assuming input images are 28x28 flattened
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)# 10 output classes (digits 0-9)
    
    def forward(self, img):
        # Flattening the image tensor
        x = img.view(img.size(0), -1)
        # Activation functions for hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x