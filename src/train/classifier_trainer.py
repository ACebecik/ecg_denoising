"""
This file implements the class Trainer for the classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, lr = 0.001,
                 scheduler = None,
                 device = None,
                 save_best_model = None,
                 early_stop_patience = 5,
                 save_path = "assets/best_model.pth"):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler

        self.save_best_model = save_best_model
        self.patience = early_stop_patience
        self.save_path = save_path

        self.best_val_acc = 0.00
        self.epochs_no_improvement = 0

    def train(self, train_loader, val_loader, no_epochs):
        for epoch in tqdm(range(no_epochs)):
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch [{epoch+1}/{no_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc>self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improvement = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                self.epochs_no_improvement += 1
                if self.epochs_no_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_one_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X = X.unsqueeze(dim=1)
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), y.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(outputs)>0.5).int()
            correct += (preds.squeeze() == y).sum().item()
            total += y.size(0)

        return total_loss/total , correct/total

    def _validate(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.unsqueeze(dim=1)
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs.squeeze(), y.float())

                total_loss += loss.item() * X.size(0)
                preds = (torch.sigmoid(outputs)> 0.5).int()
                correct += (preds.squeeze() == y).sum().item()
                total += y.size(0)

        return total_loss/total, correct/total

    def test(self, test_loader):
        test_loss , test_accuracy = self._validate(test_loader)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
        return test_loss, test_accuracy


