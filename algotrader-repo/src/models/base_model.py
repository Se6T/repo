from functools import partial

import numpy as np

from sklearn import metrics
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_metrics_dict(task_type: str = 'binary'):
    metrics_dict = {
        'binary': {
            'accuracy': metrics.accuracy_score,
            'balanced_accuracy': partial(metrics.balanced_accuracy_score, adjusted=True),
            'f1': partial(metrics.f1_score, average='binary'),
            'precision': partial(metrics.precision_score, average='binary'),
            'recall': partial(metrics.recall_score, average='binary'),
            'roc_auc': metrics.roc_auc_score,
            'zero_one_loss': metrics.zero_one_loss,
            'jaccard_score': metrics.jaccard_score,
            'hamming_loss': metrics.hamming_loss
        },
        # Add additional branches for other task types if needed
    }

    return metrics_dict[task_type]


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def _pred(self, X: torch.tensor, device: str) -> float:
        X = X.to(device)
        return self(X)
    
    @torch.no_grad()
    def _val_fwd(self, X: torch.tensor, y: torch.tensor, criterion: nn.Module, device: str = 'cuda') -> float:
        pred = self._pred(X, device=device)
        loss = criterion(pred.view(y.size()), y.to(device))
        return loss

    def _train_fwd(
            self, X: torch.tensor, y: torch.tensor, criterion: nn.Module, optimizer: optim, device: str = 'cuda'
        ) -> float:
        optimizer.zero_grad()
        pred = self._pred(X, device=device)
        loss = criterion(pred.view(y.size()), y.to(device))
        loss.backward()
        optimizer.step()
        return loss
    
    def run_epoch(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
            criterion: nn.Module, 
            optimizer: optim, 
            device: str = 'cuda'
        ) -> list[float, float]:

        train_losses, val_losses = [], []

        self.eval()
        for X, y in val_loader:
            val_loss = self._val_fwd(X, y, criterion, device)
            val_losses.append(val_loss)
        
        self.train()
        for X, y in train_loader:
            train_loss = self._train_fwd(X, y, criterion, optimizer, device)
            train_losses.append(train_loss)
        
        return sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)
    
    def run_training(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
            criterion: nn.Module, 
            optimizer: optim,
            num_epochs: int = 5,  
            device: str = 'cuda'
        ) -> tuple[list, list]:

        train_losses, val_losses = [], []
        progress_bar = tqdm(total=num_epochs, desc='Training Progress', position=0, leave=True)
        for _ in range(num_epochs):
            train_loss, val_loss = self.run_epoch(train_loader, val_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            progress_bar.set_postfix({'Train Loss': train_loss, 'Validation Loss': val_loss})
            progress_bar.update(1)
        progress_bar.close()

        return train_losses, val_losses
    
    @torch.no_grad()
    def predict_loader(
        self, val_loader: DataLoader, sigma: float = 0.75, device: str = 'cuda'
    ) -> tuple[np.array, np.array, np.array]:
        y_hat, y_true = [], []
        for X, y in val_loader:
            X = X.to(device)
            outputs = self(X).flatten().cpu().numpy()
            y_true.append(y.numpy())
            y_hat.append(outputs)
        y_true = np.concatenate(y_true)
        y_hat = np.concatenate(y_hat)
        lim = np.percentile(y_hat, sigma*100)
        y_pred = np.where(y_hat > lim, 1, 0)
        
        return y_hat, y_pred, y_true

    def _calculate_performance_metrics(
            self, y_hat: np.array, y_pred: np.array, y_true: np.array, metrics_dict: dict
        ) -> np.array:
        performance_metrics = []
        for metric_key in metrics_dict.keys():
            metric = metrics_dict[metric_key]
            if metric == metrics.roc_auc_score:
                score = metric(y_true, y_hat)
            else:
                score = metric(y_true, y_pred)
            performance_metrics.append(score)

        return performance_metrics
    
    def _print_performace_metrics(performance_metrics: list, metrics_keys: list) -> None:
        for metric_name, score in zip(metrics_keys, performance_metrics):
            print(f"{metric_name}: {score:.3f}")
        print("\n")
    
    @torch.no_grad()
    def run_validation(
        self, val_loader: DataLoader, sigma: float = 0.75, task_type: str = 'binary', device: str = 'cuda'
    ) -> None:
        y_hat, y_pred, y_true = self._predict_loader(val_loader, sigma, device)
        if self.task_type == 'binary':
            metrics_dict = get_metrics_dict(task_type)
            performance_metrics = self._calculate_performance_metrics(y_hat, y_pred, y_true, metrics_dict)
            self._print_preformance_metrics(performance_metrics)
        else:
            raise NotImplementedError(f"No other task other than binary is implemented.")

        return performance_metrics
    


    
    @torch.no_grad()
    def _test(self, val_loader, print_task_averages, sigma=0.75):
        scores = np.zeros((10,), dtype=float)
        task_counters = np.zeros(10, dtype=int)
        y_hat, y_true = [], []
        
        for X, y in val_loader:
            X = X.to('cuda')
            outputs = self(X).flatten().cpu().numpy()
            y_true.append(y.numpy())
            y_hat.append(outputs)
        
        y_true = np.concatenate(y_true)
        y_hat = np.concatenate(y_hat)
        lim = np.percentile(y_hat, sigma*100)
        
        y_pred = np.where(y_hat > lim, 1, 0)
            
        if np.sum(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
            f1 = f1_score(y_true, y_pred, average='binary')
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            roc_auc = roc_auc_score(y_true, y_hat)
            pr_score = average_precision_score(y_true, y_pred)
            zo_loss = zero_one_loss(y_true, y_pred)
            jaccard = jaccard_score(y_true, y_pred)
            hamming = hamming_loss(y_true, y_pred)
            
            metrics = [accuracy, balanced_acc, f1, precision, recall, roc_auc, pr_score, zo_loss, jaccard, hamming]
            metrics = np.array(metrics)
            scores += metrics
            task_counters += 1
        else:
            print(f"Task {i} - y_true has sum 0.")
                        
        self.task_averages = scores / task_counters
        
        if print_task_averages:
            print(f"Average Metrics for Outputs, maximumg value and expected value for randomly generated results with 25% chance of 1:")
            print(f"Accuracy: {task_averages[0]} / 1, E_random=0.625")
            print(f"Balanced Accuracy: {task_averages[1]} / 1, E_random=0.0")
            print(f"F1 Score: {task_averages[2]} / 1, E_random=0.25")
            print(f"Precision: {task_averages[3]} / 1, E_random=0.25")
            print(f"Recall: {task_averages[4]} / 1, E_random=0.25")
            print(f"ROC AUC: {task_averages[5]} / 1, E_random=0.5")
            print(f"Average Precision Score: {task_averages[6]} / 1, E_random=0.25")
            print(f"Zero-One Loss: {task_averages[7]} / 0, E_random=0.375")
            print(f"Jaccard Score: {task_averages[8]} / 1, E_random=0.14")
            print(f"Hamming Loss: {task_averages[9]} / 0, E_random=0.375")
            print()

    
    def predict(self, X, device='cuda'):
        with torch.no_grad():
            X = X.to(device)
            outputs = self(X)
            return outputs.cpu().numpy()