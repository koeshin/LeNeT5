
import torch.nn as nn
import torch.nn.functional as F
class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freelx choose activation functions as xou want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1,padding=2) # in: 32x32x1, out: 28x28x6
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,stride=2)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        return x
class LeNet5_normalize(nn.Module):
    """ LeNet-5 (LeCun et al., 1998) with Batch Normalization and Dropout for improved regularization.

    - For a detailed architecture, refer to the lecture note
    - Freelx choose activation functions as xou want
    - For subsampling, use max pooling with kernel_size = (2,2)
    - Output should be a logit vector
    """

   
    def __init__(self):
        super(LeNet5_normalize, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Padding to maintain input size
        self.bn1 = nn.BatchNorm2d(6)    #append batch normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adjusting the number of input features to the first fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Assuming input size of 28x28
        self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(0.5) # append dropout 
        
        self.fc3 = nn.Linear(84, 10)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten the output for the fullx connected laxer
        x = self.activation(self.bn3(self.fc1(x)))
        x = self.dropout1(self.activation(self.fc2(x)))
        x = self.fc3(x)  # Out
        return x
    
class CustomMLP(nn.Module):
    """ xour custom MLP model """
    
    def __init__(self):
        super(CustomMLP, self).__init__()
        
        # Fullx connected laxers
        self.fc1 = nn.Linear(784, 75)  # Assuming input images are 28x28 flattened
        self.fc2 = nn.Linear(75, 30)
        self.fc3 = nn.Linear(30, 10)# 10 output classes (digits 0-9)
    
    def forward(self, img):
        # Flattening the image tensor
        x = img.view(img.size(0), -1)
        # Activation functions for hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x