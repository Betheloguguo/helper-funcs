import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, test_loader, loss, optimizer, device = None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.optimizer = optimizer

        if device:
            self.device = device

        self.results = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': []
                }
        
    def calculate_accuracy(self, output, target):
        _, preds = torch.max(output, 1)
        correct = (preds == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy


    def training_steps(self):
        
        self.model.train()

        running_loss, running_acc = 0, 0

        for batch_idx,  (X, y) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = self.model(X)

            loss = self.loss(outputs, y)

            running_loss += loss.item() * X.size(0)
            running_acc += self.calculate_accuracy(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss / len(self.train_loader)
        train_acc = running_acc / len(self.train_loader)

        return train_loss, train_acc
    
    def testing_steps(self):

        self.model.eval()

        running_loss, running_acc = 0, 0

        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(self.test_loader):

                X, y = X.to(self.device), y.to(self.device)
            
                outputs = self.model(X)

                loss = self.loss(outputs, y)
                running_loss += loss.item() * X.size(0)

                running_acc += self.calculate_accuracy(outputs, y)

            test_loss = running_loss/len(self.test_loader)
            test_acc = running_acc/len(self.test_loader)

        return test_loss, test_acc
    
    
    def epoch_training(self, epochs: int = 5):
        for epoch in range(epochs):
            train_loss, train_acc = self.training_steps()
            test_loss, test_acc = self.training_steps()

            self.results['train_loss'].append(np.array(train_loss))
            self.results['train_acc'].append(np.array(train_acc))
            self.results['test_loss'].append(np.array(test_loss))
            self.results['test_acc'].append(np.array(test_acc))

        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

        return self.results

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

