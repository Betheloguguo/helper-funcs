import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from pathlib import Path
import zipfile
import requests
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, test_loader, loss, optimizer, scheduler = None, device = None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler


        if device:
            self.device = device

        self.results = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': []
                }
        
    def correct_preds(self, output, target):
        _, preds = torch.max(output, 1)
        correct = (preds == target).sum().item()
        return correct


    def training_steps(self):
        
        self.model.train()

        running_loss = 0 
        correct = 0
        total_samples = 0

        for batch_idx,  (X, y) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = self.model(X)

            loss = self.loss(outputs, y)

            running_loss += loss.item() * X.size(0)
            correct += self.correct_preds(outputs, y)
            total_samples += int(np.array(y.size(0)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss / len(self.train_loader.dataset)
        train_acc = correct / total_samples

        return train_loss, train_acc
    
    def testing_steps(self):

        self.model.eval()

        running_loss, correct, total_samples = 0, 0, 0

        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(self.test_loader):

                X, y = X.to(self.device), y.to(self.device)
            
                outputs = self.model(X)

                loss = self.loss(outputs, y)
                running_loss += loss.item() * X.size(0)

                correct += self.correct_preds(outputs, y)
                total_samples += int(np.array(y.size(0)))

            test_loss = running_loss/len(self.test_loader.dataset)
            test_acc = correct / total_samples

        return test_loss, test_acc
    
    
    def epoch_training(self, epochs: int = 5):
        for epoch in range(epochs):
            train_loss, train_acc = self.training_steps()
            test_loss, test_acc = self.testing_steps()

            self.results['train_loss'].append(np.array(train_loss))
            self.results['train_acc'].append(np.array(train_acc))
            self.results['test_loss'].append(np.array(test_loss))
            self.results['test_acc'].append(np.array(test_acc))

        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

        # ---- Step scheduler ----
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()


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



def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
