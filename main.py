
import dataset
from model import LeNet5, CustomMLP

# import some packages you need here
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
from model import LeNet5, CustomMLP,LeNet5_normalize
import numpy as np
import torch.nn.functional as F
def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(trn_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    trn_loss = total_loss / len(trn_loader)
    acc = 100. * correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    tst_loss = total_loss / len(tst_loader)
    acc = 100. * correct / total
    return tst_loss, acc
def plot_training_statistics(train_losses, train_accuracies, test_losses, test_accuracies,save_path):
    """
    Plots the training and testing loss and accuracy.

    Args:
    train_losses (list): List of loss values during training.
    train_accuracies (list): List of accuracy values during training.
    test_losses (list): List of loss values during testing.
    test_accuracies (list): List of accuracy values during testing.
    """
    plt.figure(figsize=(10, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = dataset.MNIST(data_dir='../data/train')
    test_dataset = dataset.MNIST(data_dir='../data/test')

    # DataLoader
    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # print("###################### LeNet5 #####################")
    epochs = 10  # 훈련할 에폭 수
        # Dataset
    # Model LeNet5
    model = LeNet5().to(device)  # Change to CustomMLP() if you want to use the MLP model instead

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    LeNet_train_avg_loss = []
    LeNet_train_avg_acc=[]
    LeNet_test_avg_loss = []
    LeNet_test_avg_acc=[]
    for epoch in range(epochs):
        # Training and Testing
        train_loss, train_accuracy = train(model, trn_loader, device, criterion, optimizer)
        LeNet_train_avg_loss.append(train_loss)
        LeNet_train_avg_acc.append(train_accuracy)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%',flush=True)

        test_loss, test_accuracy = test(model, tst_loader, device, criterion)
        LeNet_test_avg_loss.append(test_loss)
        LeNet_test_avg_acc.append(test_accuracy)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%',flush=True)
    train_avg_loss=np.mean(LeNet_train_avg_loss)
    train_avg_acc=np.mean(LeNet_train_avg_acc)
    test_avg_loss=np.mean(LeNet_test_avg_loss)
    test_avg_acc=np.mean(LeNet_test_avg_acc)
    print(f"Train_avg_Loss:{train_avg_loss:.4f},Train_avg_Accuracy:{train_avg_acc:.2f},Test_avg_Loss:{test_avg_loss:.4f},Test_avg_Accuracy:{test_avg_acc:.2f}%",flush=True)
    

    # write your codes here
    print("###################### Custom MLP #####################")
    # Model-CustomMLP
    model2 = CustomMLP().to(device)

    # Optimizer
    optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    CustomMLP_train_avg_loss = []
    CustomMLP_train_avg_acc=[]
    CustomMLP_test_avg_loss = []
    CustomMLP_test_avg_acc=[]
    for epoch in range(epochs):
    # Training and Testing
        train_loss, train_accuracy = train(model2, trn_loader, device, criterion, optimizer)
        CustomMLP_train_avg_loss.append(train_loss)
        CustomMLP_train_avg_acc.append(train_accuracy)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%',flush=True)

        test_loss, test_accuracy = test(model2, tst_loader, device, criterion)
        CustomMLP_test_avg_loss.append(test_loss)
        CustomMLP_test_avg_acc.append(test_accuracy)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%',flush=True)

    train_avg_loss=np.mean(CustomMLP_train_avg_loss)
    train_avg_acc=np.mean(CustomMLP_train_avg_acc)
    test_avg_loss=np.mean(CustomMLP_test_avg_loss)
    test_avg_acc=np.mean(CustomMLP_test_avg_acc)
    print(f"Train_avg_Loss:{train_avg_loss:.4f},Train_avg_Accuracy:{train_avg_acc:.2f},Test_avg_Loss:{test_avg_loss:.4f},Test_avg_Accuracy:{test_avg_acc:.2f}%",flush=True)
    
    
    # plot_training_statistics(LeNet_train_avg_loss,LeNet_train_avg_acc,LeNet_test_avg_loss,LeNet_test_avg_acc,"LeNet5_normalize_plot_epoch=10")
    # print(CustomMLP_train_avg_loss)
    plot_training_statistics(CustomMLP_train_avg_loss,CustomMLP_train_avg_acc,CustomMLP_test_avg_loss,CustomMLP_test_avg_acc,"CustumMLP_plot_epoch=10")

def main2():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = dataset.MNIST(data_dir='../data/train',augment=True)
    test_dataset = dataset.MNIST(data_dir='../data/test',augment=False)

    # DataLoader
    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # print("###################### LeNet5 #####################")
    epochs = 10  # 훈련할 에폭 수
        # Dataset
    # Model LeNet5
    model = LeNet5().to(device)  # Change to CustomMLP() if you want to use the MLP model instead

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    LeNet_train_avg_loss = []
    LeNet_train_avg_acc=[]
    LeNet_test_avg_loss = []
    LeNet_test_avg_acc=[]
    for epoch in range(epochs):
        # Training and Testing
        train_loss, train_accuracy = train(model, trn_loader, device, criterion, optimizer)
        LeNet_train_avg_loss.append(train_loss)
        LeNet_train_avg_acc.append(train_accuracy)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%',flush=True)

        test_loss, test_accuracy = test(model, tst_loader, device, criterion)
        LeNet_test_avg_loss.append(test_loss)
        LeNet_test_avg_acc.append(test_accuracy)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%',flush=True)
    train_avg_loss=np.mean(LeNet_train_avg_loss)
    train_avg_acc=np.mean(LeNet_train_avg_acc)
    test_avg_loss=np.mean(LeNet_test_avg_loss)
    test_avg_acc=np.mean(LeNet_test_avg_acc)
    print(f"Train_avg_Loss:{train_avg_loss:.4f},Train_avg_Accuracy:{train_avg_acc:.2f},Test_avg_Loss:{test_avg_loss:.4f},Test_avg_Accuracy:{test_avg_acc:.2f}%",flush=True)

    plot_training_statistics(LeNet_train_avg_loss,LeNet_train_avg_acc,LeNet_test_avg_loss,LeNet_test_avg_acc,"LeNet5_normalize_plot_epoch=10")

if __name__ == '__main__':
    # main()
    main2()
