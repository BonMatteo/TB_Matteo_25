import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from preprocess import BaseflowDataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import json
from datetime import datetime

# Get the absolute path to the DNN directory
base_dir = os.path.dirname(os.path.abspath(__file__))

class LocalMetricsLogger:
    """Logger for storing metrics locally in a structured format.
    
    This logger mirrors the WandB logging structure but stores data locally in CSV files.
    Each run gets its own directory with separate files for different metric types.
    """
    def __init__(self, run_id, save_dir="local_metrics"):
        # Get the absolute path to the test-NN directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, save_dir, run_id)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = {
            'train': {},
            'val': {},
            'test': {},
            'catchment_metrics': {},
            'config': {}
        }
        
        # Create files for each metric type
        self._init_metric_files()
        
        # Store catchment metrics until epoch end
        self.current_epoch_catchment_metrics = {
            'training': {},
            'validation': {},
            'test': {}
        }
    
    def _init_metric_files(self):
        """Initialize CSV files for each metric type"""
        # For training/validation/test metrics
        for metric_type in ['train', 'val', 'test']:
            file_path = os.path.join(self.save_dir, f"{metric_type}_metrics.csv")
            # Create empty file with headers
            with open(file_path, 'w') as f:
                f.write("epoch,step,loss,learning_rate\n")
    
    def log_metrics(self, metrics_dict, step=None, metric_type='train', epoch=None):
        """Log metrics to local storage
        
        Args:
            metrics_dict: Dictionary of metrics to log
            step: Current step number
            metric_type: Type of metrics ('train', 'val', 'test', 'catchment_metrics')
            epoch: Current epoch number
        """
        if metric_type not in self.metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        if metric_type in ['train', 'val', 'test']:
            # Handle training/validation/test metrics
            file_path = os.path.join(self.save_dir, f"{metric_type}_metrics.csv")
            
            # Extract metrics
            loss = metrics_dict.get(f'{metric_type}/loss', '')
            lr = metrics_dict.get(f'{metric_type}/learning_rate', '')
            
            # Write to CSV
            with open(file_path, 'a') as f:
                f.write(f"{epoch},{step},{loss},{lr}\n")
                
        elif metric_type == 'catchment_metrics':
            # Handle catchment metrics
            for metric_name, metric_value in metrics_dict.items():
                # Parse metric name (e.g., "training_catchments/Sense/mse")
                parts = metric_name.split('/')
                if len(parts) == 3:
                    phase, catchment, metric = parts
                    # Convert phase name (e.g., "training_catchments" -> "training")
                    phase = phase.replace('_catchments', '')
                    
                    if phase not in self.current_epoch_catchment_metrics:
                        self.current_epoch_catchment_metrics[phase] = {}
                    if catchment not in self.current_epoch_catchment_metrics[phase]:
                        self.current_epoch_catchment_metrics[phase][catchment] = {}
                    
                    self.current_epoch_catchment_metrics[phase][catchment][metric] = metric_value
    
    def on_epoch_end(self, epoch):
        """Write all collected catchment metrics at the end of each epoch"""
        for phase, catchments in self.current_epoch_catchment_metrics.items():
            if catchments:  # Only write if we have metrics
                file_path = os.path.join(self.save_dir, f"{phase}_catchment_metrics_epoch_{epoch}.csv")
                
                # Write header and data
                with open(file_path, 'w') as f:
                    # Write header
                    f.write("catchment,mse,mae,rmse,r2,kge\n")
                    
                    # Write data for each catchment
                    for catchment, metrics in catchments.items():
                        metrics_str = ','.join(str(metrics.get(m, '')) for m in ['mse', 'mae', 'rmse', 'r2', 'kge'])
                        f.write(f"{catchment},{metrics_str}\n")
        
        # Reset the metrics collection for next epoch
        self.current_epoch_catchment_metrics = {
            'training': {},
            'validation': {},
            'test': {}
        }
    
    def log_config(self, config_dict):
        """Log configuration parameters
        
        Args:
            config_dict: Dictionary of configuration parameters
        """
        # Convert numpy types to Python native types
        config_dict = {
            k: v.item() if hasattr(v, 'item') else v 
            for k, v in config_dict.items()
        }
        
        file_path = os.path.join(self.save_dir, "config.csv")
        df = pd.DataFrame([config_dict])
        df.to_csv(file_path, index=False)

class WarmupDecayScheduler:
    """Learning rate scheduler with three phases:
    1. Warmup: Linear increase from init_lr to peak_lr
    2. Constant: Maintains peak_lr
    3. Decay: Linear decrease from peak_lr to final_lr
    """
    def __init__(self, 
                 optimizer,
                 init_lr,
                 peak_lr,
                 final_lr,
                 warmup_steps,
                 constant_steps,
                 decay_steps):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        current_lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        return current_lr
    
    def _get_lr(self):
        # During warmup phase: linear increase
        if self.current_step <= self.warmup_steps:
            return self.init_lr + (self.peak_lr - self.init_lr) * (self.current_step / self.warmup_steps)
        
        # During constant phase: maintain peak learning rate
        elif self.current_step <= (self.warmup_steps + self.constant_steps):
            return self.peak_lr
        
        # During decay phase: linear decrease
        elif self.current_step <= (self.warmup_steps + self.constant_steps + self.decay_steps):
            decay_progress = (self.current_step - self.warmup_steps - self.constant_steps) / self.decay_steps
            return self.peak_lr + (self.final_lr - self.peak_lr) * decay_progress
        
        # After all phases: maintain final learning rate
        else:
            return self.final_lr

class Block(nn.Module):
    """Residual block with multi-layer processing path and skip connection.
    Architecture: 
        1. Input → Multiple (Linear → Dropout → ReLU) layers
        2. Layer normalization of processed features
        3. Addition of processed features with transformed input (residual connection)
        4. Final ReLU activation
    """
    def __init__(self, in_features, out_features, layers_per_block, dropout_rate):
        super(Block, self).__init__()
        # Transform residual connection if dimensions don't match
        self.residual = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features)
        ) if in_features != out_features else nn.Identity()
        
        # Main processing path
        block_layers = []
        for _ in range(layers_per_block):
            block_layers.extend([
                nn.Linear(out_features, out_features, bias=True),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
            ])
        block_layers.append(nn.LayerNorm(out_features))
        self.block = nn.Sequential(*block_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass combining main path with residual connection
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Processed tensor after residual addition and activation
        """
        residual = self.residual(x)
        out = self.block(x)
        return self.relu(out + residual)

class CatchmentMetricsTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, catchment, mse, y_pred, y_true):
        if catchment not in self.metrics:
            self.metrics[catchment] = {'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'kge': []}
        
        # Calculate metrics using numpy
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to prevent division by zero
        
        # Calculate Kling-Gupta Efficiency (KGE)
        r = np.corrcoef(y_true, y_pred)[0, 1]  # Correlation coefficient
        alpha = np.std(y_pred) / (np.std(y_true) + 1e-10)  # Ratio of standard deviations
        beta = np.mean(y_pred) / (np.mean(y_true) + 1e-10)  # Ratio of means
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        # Store metrics
        self.metrics[catchment]['mse'].append(mse)
        self.metrics[catchment]['mae'].append(mae)
        self.metrics[catchment]['rmse'].append(rmse)
        self.metrics[catchment]['r2'].append(r2)
        self.metrics[catchment]['kge'].append(kge)

    def get_summary(self):
        summary = {}
        for catchment, metrics in self.metrics.items():
            summary[catchment] = {
                'mse': np.mean(metrics['mse']),
                'mae': np.mean(metrics['mae']),
                'rmse': np.mean(metrics['rmse']),
                'r2': np.mean(metrics['r2']),
                'kge': np.mean(metrics['kge'])
            }
        return pd.DataFrame.from_dict(summary, orient='index')

    def plot_performance_comparison(self):
        fig, ax = plt.subplots(3, 2, figsize=(15, 15))
        metrics_df = self.get_summary()
        
        # Sort catchments alphabetically
        metrics_df = metrics_df.sort_index()  # Sort by catchment names
        
        # Plot MSE
        ax[0,0].bar(range(len(metrics_df)), metrics_df['mse'])
        ax[0,0].set_title('Mean Squared Error')
        ax[0,0].set_ylabel('MSE')
        ax[0,0].set_xticks(range(len(metrics_df)))
        ax[0,0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Plot MAE
        ax[0,1].bar(range(len(metrics_df)), metrics_df['mae'])
        ax[0,1].set_title('Mean Absolute Error')
        ax[0,1].set_ylabel('MAE')
        ax[0,1].set_xticks(range(len(metrics_df)))
        ax[0,1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Plot RMSE
        ax[1,0].bar(range(len(metrics_df)), metrics_df['rmse'])
        ax[1,0].set_title('Root Mean Squared Error')
        ax[1,0].set_ylabel('RMSE')
        ax[1,0].set_xticks(range(len(metrics_df)))
        ax[1,0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Plot R²
        ax[1,1].bar(range(len(metrics_df)), metrics_df['r2'])
        ax[1,1].set_title('R² Score')
        ax[1,1].set_ylabel('R²')
        ax[1,1].set_xticks(range(len(metrics_df)))
        ax[1,1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Plot KGE
        ax[2,0].bar(range(len(metrics_df)), metrics_df['kge'])
        ax[2,0].set_title('Kling-Gupta Efficiency')
        ax[2,0].set_ylabel('KGE')
        ax[2,0].set_xticks(range(len(metrics_df)))
        ax[2,0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Add grid for better readability
        for a in ax.flat:
            a.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_kge_comparison(self):
        """Create a separate visualization focused on KGE metrics only"""
        fig = plt.figure(figsize=(12, 6))
        metrics_df = self.get_summary()
        
        # Sort catchments alphabetically
        metrics_df = metrics_df.sort_index()
        
        # Plot KGE values as a bar chart
        plt.bar(range(len(metrics_df)), metrics_df['kge'])
        plt.title('Kling-Gupta Efficiency by Catchment')
        plt.ylabel('KGE Score')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.7, label='KGE = 0.5')
        plt.axhline(y=0.7, color='g', linestyle='-', alpha=0.7, label='KGE = 0.7')
        plt.legend()
        
        # Add catchment labels
        plt.xticks(range(len(metrics_df)), metrics_df.index, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig

class BaseflowNN(pl.LightningModule):
    """Neural network for baseflow prediction with residual architecture.
    
    Architecture:
    - Input normalization 
    - Initial feature transformation
    - Multiple residual blocks for deep feature processing
    - Final output layer producing 366 daily baseflow values
    
    Includes learning rate scheduling, metrics tracking by catchment,
    and comprehensive logging functionality.
    """
    def __init__(self, 
                 layer_dimensions,
                 dropout_rate,
                 init_lr,
                 peak_lr,
                 final_lr,
                 warmup_epochs,
                 constant_epochs,
                 decay_epochs,
                 full_data,
                 batch_size,
                 num_blocks=5,
                 layers_per_block=2,
                 split_method='random',
                 test=False):
        super().__init__()
        self.save_hyperparameters()
        # Log config and input shape
        print("\n[CONFIG] Model configuration:")
        print(f"  full_data: {full_data}")
        print(f"  input_features: {['Q', 'Pmean', 'Tmean'] if full_data else ['Q']}")
        input_features = 366 * 3 if full_data else 366
        print(f"  input_shape: {input_features}")
        print(f"  layer_dimensions: {layer_dimensions}")
        print(f"  num_blocks: {num_blocks}, layers_per_block: {layers_per_block}")
        print(f"  split_method: {split_method}, test: {test}")
        print(f"  batch_size: {batch_size}")
        print(f"  learning rates: init={init_lr}, peak={peak_lr}, final={final_lr}")
        
        # Initialize catchment metrics tracker
        self.validation_catchment_tracker = CatchmentMetricsTracker()
        self.training_catchment_tracker = CatchmentMetricsTracker()
        self.test_catchment_tracker = CatchmentMetricsTracker()
        
        # Input size depends on whether we use only flow or all features
        input_features = 366 * 3 if full_data else 366
        
        # Network construction
        layers = []
        # 1. Input processing
        layers.append(nn.LayerNorm(input_features))
        layers.append(nn.Linear(input_features, layer_dimensions[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 2. Processing blocks
        for b in range(num_blocks):
            in_features = layer_dimensions[0] if b == 0 else layer_dimensions[-1]
            out_features = layer_dimensions[-1]
            layers.append(Block(in_features, out_features, layers_per_block, dropout_rate))
        
        # 3. Output processing
        layers.append(nn.LayerNorm(layer_dimensions[-1]))
        layers.append(nn.Linear(layer_dimensions[-1], 366))
        self.network = nn.Sequential(*layers)
        
        # Initialize biases for non-residual linear layers
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        
        self.criterion = nn.MSELoss()
        
        # Initialize learning rate tracking
        self.current_lr = init_lr
        
        # Storage for performance tracking
        self.prediction_performances = []
        self.training_prediction_performances = []
        self.test_prediction_performances = []
        
        # Get the WandB run ID if available
        self.run_id = wandb.run.id if wandb.run else "no_wandb"
        
        # Initialize local metrics logger
        self.local_logger = LocalMetricsLogger(self.run_id)
        
        # Log initial configuration
        config_dict = {
            'layer_dimensions': layer_dimensions,
            'dropout_rate': dropout_rate,
            'init_lr': init_lr,
            'peak_lr': peak_lr,
            'final_lr': final_lr,
            'warmup_epochs': warmup_epochs,
            'constant_epochs': constant_epochs,
            'decay_epochs': decay_epochs,
            'full_data': full_data,
            'batch_size': batch_size,
            'num_blocks': num_blocks,
            'layers_per_block': layers_per_block,
            'split_method': split_method,
            'test': test
        }
        self.local_logger.log_config(config_dict)
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y, catchment_names = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Track metrics for each catchment sample during training
        for i in range(len(catchment_names)):
            sample_mse = torch.mean((y_hat[i] - y[i])**2).item()
            
            self.training_catchment_tracker.update(
                catchment_names[i],
                sample_mse,
                y_hat[i].detach().cpu().numpy(),
                y[i].cpu().numpy()
            )
            
            self.training_prediction_performances.append({
                'mse': sample_mse,
                'prediction': y_hat[i].detach().cpu(),
                'target': y[i].cpu(),
                'catchment': catchment_names[i]
            })
        
        # Log training metrics
        metrics = {
            'train/loss': loss.item(),
            'train/learning_rate': self.current_lr
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics locally
        self.local_logger.log_metrics(metrics, step=self.global_step, metric_type='train', epoch=self.current_epoch)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, catchment_names = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        
        # Track metrics for each catchment sample
        for i in range(len(catchment_names)):
            sample_mse = torch.mean((y_hat[i] - y[i])**2).item()
            
            self.validation_catchment_tracker.update(
                catchment_names[i],
                sample_mse,
                y_hat[i].detach().cpu().numpy(),
                y[i].cpu().numpy()
            )
            
            self.prediction_performances.append({
                'mse': sample_mse,
                'prediction': y_hat[i].detach().cpu(),
                'target': y[i].cpu(),
                'catchment': catchment_names[i]
            })
        
        # Log validation metrics
        metrics = {
            'val/loss': val_loss.item()
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics locally
        self.local_logger.log_metrics(metrics, step=self.global_step, metric_type='val', epoch=self.current_epoch)
        
        return val_loss
    
    def on_train_epoch_end(self):
        """Log visualization of catchment metrics at training epoch end"""
        if len(self.training_prediction_performances) > 0:
            epoch = self.current_epoch
            
            # Log catchment-specific metrics
            metrics_df = self.training_catchment_tracker.get_summary()
            
            if self.logger:
                # Log individual catchment metrics
                for catchment in metrics_df.index:
                    metrics = {
                        f"training_catchments/{catchment}/mse": metrics_df.loc[catchment, 'mse'],
                        f"training_catchments/{catchment}/mae": metrics_df.loc[catchment, 'mae'],
                        f"training_catchments/{catchment}/rmse": metrics_df.loc[catchment, 'rmse'],
                        f"training_catchments/{catchment}/r2": metrics_df.loc[catchment, 'r2'],
                        f"training_catchments/{catchment}/kge": metrics_df.loc[catchment, 'kge'],
                    }
                    self.log_dict(metrics, sync_dist=True)
                    
                    # Log metrics locally
                    self.local_logger.log_metrics(
                        metrics,
                        step=epoch,
                        metric_type='catchment_metrics',
                        epoch=epoch
                    )
                
                # Log performance comparison plot
                fig = self.training_catchment_tracker.plot_performance_comparison()
                self.logger.experiment.log({
                    f"train_catchment_metrics/epoch_{epoch}/comparison": wandb.Image(fig),
                    "global_step": self.global_step
                })
                plt.close(fig)
                
                # Log KGE comparison plot
                kge_fig = self.training_catchment_tracker.plot_kge_comparison()
                self.logger.experiment.log({
                    f"train_catchment_metrics/epoch_{epoch}/kge_comparison": wandb.Image(kge_fig),
                    "global_step": self.global_step
                })
                plt.close(kge_fig)
            
            # Write all collected metrics to files
            self.local_logger.on_epoch_end(epoch)
            
            # Reset tracker for next epoch
            self.training_prediction_performances = []
            self.training_catchment_tracker = CatchmentMetricsTracker()
    
    def on_validation_epoch_end(self):
        """Log visualization of best, worst, and average predictions at validation end"""
        if len(self.prediction_performances) > 0:
            # Calculate KGE for each prediction
            for pred_data in self.prediction_performances:
                y_true = pred_data['target'].cpu().numpy()
                y_pred = pred_data['prediction'].detach().cpu().numpy()
                
                # Calculate KGE components
                r = np.corrcoef(y_true, y_pred)[0, 1]
                alpha = np.std(y_pred) / (np.std(y_true) + 1e-10)
                beta = np.mean(y_pred) / (np.mean(y_true) + 1e-10)
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                # Add KGE to prediction data
                pred_data['kge'] = kge
            
            # Sort predictions by KGE
            sorted_predictions = sorted(self.prediction_performances, key=lambda x: x['kge'], reverse=True)
            
            # Get total number of predictions
            n_predictions = len(sorted_predictions)
            
            # Select best, worst, and average predictions to log
            predictions_to_log = {
                'best': sorted_predictions[:3],  # Highest KGE
                'worst': sorted_predictions[-3:],  # Lowest KGE
                'average': sorted_predictions[n_predictions//2-1:n_predictions//2+2]  # Middle KGE
            }
            
            # Log selected predictions
            epoch = self.current_epoch
            for category, predictions in predictions_to_log.items():
                for idx, pred_data in enumerate(predictions):
                    self._log_predictions(
                        pred_data['target'],
                        pred_data['prediction'],
                        pred_data['catchment'],
                        f'{category}/{pred_data["catchment"]}'
                    )
            
            # Create and log MSE distribution plot
            fig = plt.figure(figsize=(10, 5))
            mse_values = [p['mse'] for p in self.prediction_performances]
            plt.hist(mse_values, bins=50)
            plt.title('Distribution of Prediction MSE')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Count')
            self.logger.experiment.log({
                f"val_distributions/epoch_{epoch}/mse_distribution": wandb.Image(fig),
                "global_step": self.global_step
            })
            plt.close()
            
            # Log catchment-specific metrics
            metrics_df = self.validation_catchment_tracker.get_summary()
            
            if self.logger:
                # Log individual catchment metrics
                for catchment in metrics_df.index:
                    metrics = {
                        f"validation_catchments/{catchment}/mse": metrics_df.loc[catchment, 'mse'],
                        f"validation_catchments/{catchment}/mae": metrics_df.loc[catchment, 'mae'],
                        f"validation_catchments/{catchment}/rmse": metrics_df.loc[catchment, 'rmse'],
                        f"validation_catchments/{catchment}/r2": metrics_df.loc[catchment, 'r2'],
                        f"validation_catchments/{catchment}/kge": metrics_df.loc[catchment, 'kge'],
                    }
                    self.log_dict(metrics, sync_dist=True)
                    
                    # Log metrics locally
                    self.local_logger.log_metrics(
                        metrics,
                        step=epoch,
                        metric_type='catchment_metrics',
                        epoch=epoch
                    )
                
                # Log performance comparison plot
                fig = self.validation_catchment_tracker.plot_performance_comparison()
                self.logger.experiment.log({
                    f"val_catchment_metrics/epoch_{epoch}/comparison": wandb.Image(fig),
                    "global_step": self.global_step
                })
                plt.close(fig)
                
                # Log KGE comparison plot
                kge_fig = self.validation_catchment_tracker.plot_kge_comparison()
                self.logger.experiment.log({
                    f"val_catchment_metrics/epoch_{epoch}/kge_comparison": wandb.Image(kge_fig),
                    "global_step": self.global_step
                })
                plt.close(kge_fig)
            
            # Write all collected metrics to files
            self.local_logger.on_epoch_end(epoch)
            
            # Reset trackers for next epoch
            self.prediction_performances = []
            self.validation_catchment_tracker = CatchmentMetricsTracker()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update learning rate at each step only during training
        if self.trainer.training and hasattr(self, 'lr_scheduler'):
            self.current_lr = self.lr_scheduler.step()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler
        
        Returns:
            optimizer: AdamW optimizer with weight decay
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.init_lr,
            weight_decay=0.01
        )
        
        # Calculate total steps for each learning rate phase
        steps_per_epoch = len(self.train_dataloader())
        warmup_steps = steps_per_epoch * self.hparams.warmup_epochs
        constant_steps = steps_per_epoch * self.hparams.constant_epochs
        decay_steps = steps_per_epoch * self.hparams.decay_epochs
        
        self.lr_scheduler = WarmupDecayScheduler(
            optimizer=optimizer,
            init_lr=self.hparams.init_lr,
            peak_lr=self.hparams.peak_lr,
            final_lr=self.hparams.final_lr,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            decay_steps=decay_steps
        )
        
        return optimizer
    
    def _log_predictions(self, y_true, y_pred, catchment_full_name, stage='val'):
        """Create visualization comparing predicted vs actual baseflow values
        Args:
            y_true: Ground truth baseflow values
            y_pred: Model predictions
            catchment_full_name: Full catchment name (e.g., Muota_26_abcdef...)
            stage: Identifier for the visualization in logging (format: 'phase/category/catchment')
        """
        if self.logger:
            fig = plt.figure(figsize=(10, 5))
            # Extract catchment ID (e.g., Muota_26)
            if '_' in catchment_full_name:
                catchment_id = '_'.join(catchment_full_name.split('_')[:2])
            else:
                catchment_id = catchment_full_name
            # Get current epoch
            epoch = self.current_epoch
            days = range(1, 367)
            plt.plot(days, y_true.cpu().numpy(), 
                    label='Actual', alpha=0.7, color='blue')
            plt.plot(days, y_pred.detach().cpu().numpy(), 
                    label='Predicted', alpha=0.7, color='red')
            # Calculate metrics for this prediction
            mse = torch.mean((y_pred - y_true) ** 2).item()
            mae = torch.mean(torch.abs(y_pred - y_true)).item()
            # Calculate KGE
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            r = np.corrcoef(y_true_np, y_pred_np)[0, 1]
            alpha = np.std(y_pred_np) / (np.std(y_true_np) + 1e-10)
            beta = np.mean(y_pred_np) / (np.mean(y_true_np) + 1e-10)
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            plt.legend()
            plt.title(f'Daily Baseflow Prediction - {catchment_id}, epoch {epoch}\nMSE: {mse:.4f}, MAE: {mae:.4f}, KGE: {kge:.4f}')
            plt.xlabel('Day of Year')
            plt.ylabel('Baseflow (m³/s)')
            plt.grid(True, alpha=0.3)
            # Log to WandB with catchment_id in the key
            self.logger.experiment.log({
                f"predictions/epoch_{epoch}/{stage}_{catchment_id}": wandb.Image(fig),
                "global_step": self.global_step
            })
            plt.close()
    
    def prepare_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "baseflow_23x100catchments_29years")
        cache_dir = os.path.join(os.path.dirname(__file__), "preprocessed_data")
        
        print(f"\nLooking for CSV files in: {data_dir}")
        self.csv_files = [f for f in os.listdir(data_dir) if f.startswith('baseflow_') and f.endswith('.csv')]
        print(f"Found {len(self.csv_files)} CSV files")
        self.datasets = []
        
        for csv_file in tqdm(self.csv_files, desc="Loading datasets"):
            dataset = BaseflowDataset(
                os.path.join(data_dir, csv_file),
                cache_dir=cache_dir,
                full_data=self.hparams.full_data
            )
            self.datasets.append(dataset)
    
    def setup(self, stage=None):
        if not hasattr(self, 'datasets'):
            self.prepare_data()
        
        # Get split method from hyperparameters
        split_method = self.hparams.split_method
        test_mode = self.hparams.test
        
        print(f"\n{'='*80}")
        print(f"Setting up model with '{split_method}' splitting method")
        if test_mode:
            print(f"Test mode enabled: Creating train (70%), validation (20%), and test (10%) datasets")
        else:
            print(f"Test mode disabled: Creating train (80%) and validation (20%) datasets")
        print(f"{'='*80}\n")
        
        # Filter out invalid datasets (those that failed BFI check)
        valid_datasets = [dataset for dataset in self.datasets if dataset.is_valid]
        if not valid_datasets:
            raise ValueError("No valid datasets found after BFI filtering")
        
        print(f"Found {len(valid_datasets)} valid datasets out of {len(self.datasets)} total")
        
        if test_mode:
            # Three-way split: Test (10%), Validation (20% of remaining), Training (remaining)
            self._setup_three_way_split(valid_datasets, split_method)
        else:
            # Regular two-way split: Training (80%), Validation (20%)
            self._setup_two_way_split(valid_datasets, split_method)
        
        # Store as class attribute for use in training
        self.train_catchments = self.train_catchment_names
    
    def _setup_two_way_split(self, valid_datasets, split_method):
        """Split valid datasets into train and validation sets (original implementation)"""
        n_val = int(len(valid_datasets) * 0.2)  # 20% for validation
        
        if split_method == 'random':
            # Use fixed seed for reproducibility
            rng = torch.Generator().manual_seed(42)
            train_indices, val_indices = random_split(
                range(len(valid_datasets)), 
                [len(valid_datasets) - n_val, n_val],
                generator=rng
            )
            
            # Process datasets based on indices
            train_X, train_y, train_catchments = [], [], []
            val_X, val_y, val_catchments = [], [], []
            
            # Process training datasets
            self.train_catchment_names = set()
            for idx in train_indices:
                dataset = valid_datasets[idx]
                train_X.extend(dataset.X)
                train_y.extend(dataset.y)
                train_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
                self.train_catchment_names.add(dataset.base_catchment_name)
            
            # Process validation datasets
            self.val_catchment_names = set()
            for idx in val_indices:
                dataset = valid_datasets[idx]
                val_X.extend(dataset.X)
                val_y.extend(dataset.y)
                val_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
                self.val_catchment_names.add(dataset.base_catchment_name)
                
            # Print catchment information
            print(f"\nRandom splitting:")
            print(f"  Training catchments ({len(self.train_catchment_names)}): {', '.join(sorted(self.train_catchment_names))}")
            print(f"  Validation catchments ({len(self.val_catchment_names)}): {', '.join(sorted(self.val_catchment_names))}")
                
        elif split_method == 'catchment':
            # Group valid datasets by base catchment (without iteration number)
            catchment_groups = {}
            for idx, dataset in enumerate(valid_datasets):
                # Extract base catchment name (everything before the last underscore)
                base_catchment = dataset.base_catchment_name
                if base_catchment not in catchment_groups:
                    catchment_groups[base_catchment] = []
                catchment_groups[base_catchment].append(idx)
            
            # Randomly select catchments for validation
            catchments = list(catchment_groups.keys())
            n_val_catchments = max(1, int(len(catchments) * 0.2))  # At least 1 catchment for validation
            rng = torch.Generator().manual_seed(42)
            val_catchment_indices = torch.randperm(len(catchments), generator=rng)[:n_val_catchments].tolist()
            self.val_catchment_names = {catchments[i] for i in val_catchment_indices}
            self.train_catchment_names = {c for c in catchments if c not in self.val_catchment_names}
            
            # Split datasets based on catchment groups
            train_X, train_y, train_catchments = [], [], []
            val_X, val_y, val_catchments = [], [], []
            
            # Print catchment information
            print(f"\nCatchment-based splitting:")
            print(f"  Training catchments ({len(self.train_catchment_names)}): {', '.join(sorted(self.train_catchment_names))}")
            print(f"  Validation catchments ({len(self.val_catchment_names)}): {', '.join(sorted(self.val_catchment_names))}")
            
            for base_catchment, indices in catchment_groups.items():
                if base_catchment in self.val_catchment_names:
                    # Add all iterations and years from this catchment to validation
                    for idx in indices:
                        dataset = valid_datasets[idx]
                        val_X.extend(dataset.X)
                        val_y.extend(dataset.y)
                        val_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
                else:
                    # Add all iterations and years from this catchment to training
                    for idx in indices:
                        dataset = valid_datasets[idx]
                        train_X.extend(dataset.X)
                        train_y.extend(dataset.y)
                        train_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        # Convert to tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        val_X = torch.tensor(val_X, dtype=torch.float32)
        val_y = torch.tensor(val_y, dtype=torch.float32)
         
        # Create datasets with catchment names
        self.train_dataset = [(x, y, c) for x, y, c in zip(train_X, train_y, train_catchments)]
        self.val_dataset = [(x, y, c) for x, y, c in zip(val_X, val_y, val_catchments)]
        self.test_dataset = None  # No test dataset in two-way split
        
        print(f"\n[SPLIT] Two-way split summary:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Training catchments: {sorted(self.train_catchment_names)}")
        print(f"  Validation catchments: {sorted(self.val_catchment_names)}")
        
        # Verify split integrity
        print("\nVerifying split integrity...")
        
        if split_method == 'random':
            # For random splitting, check different things than for catchment-based splitting
            print("  Checking random splitting integrity:")
            
            # Check 1: Ensure we have data in both splits
            print(f"  ✓ Training set has {len(self.train_catchment_names)} unique catchments")
            print(f"  ✓ Validation set has {len(self.val_catchment_names)} unique catchments")
            
            # Check 2: It's OK for iterations of the same catchment to be in different splits for random
            # So instead, let's just report how many catchments have iterations in both splits
            # Get base catchment names (without iteration numbers)
            train_base_names = {c.split('_')[0] if '_' in c else c for c in self.train_catchment_names}
            val_base_names = {c.split('_')[0] if '_' in c else c for c in self.val_catchment_names}
            
            # Find base catchments that appear in both splits
            shared_base_catchments = train_base_names.intersection(val_base_names)
            
            print(f"  ℹ Random splitting: {len(shared_base_catchments)} catchments have iterations in both training and validation sets")
            if shared_base_catchments:
                print(f"    These include: {', '.join(sorted(shared_base_catchments))}")
        
        else:  # catchment-based splitting
            # Check 1: Ensure no dataset is in both training and validation
            train_val_overlap = self.train_catchment_names.intersection(self.val_catchment_names)
            if train_val_overlap:
                print(f"  WARNING: Found {len(train_val_overlap)} catchments in both training and validation: {', '.join(train_val_overlap)}")
            else:
                print(f"  ✓ No overlap between training and validation datasets")
            
            # Check 2: For catchment-based splitting, ensure all iterations of a catchment are in the same split
            # Get all unique catchment base names (without iteration numbers)
            all_catchments = {d.base_catchment_name for d in self.datasets if d.is_valid}
            all_base_names = {c.split('_')[0] if '_' in c else c for c in all_catchments}
            
            # Check if any base catchment has iterations in different splits
            split_issues = []
            for base_name in all_base_names:
                # Find all catchments starting with this base name
                train_iterations = [c for c in self.train_catchment_names if c.startswith(base_name + '_') or c == base_name]
                val_iterations = [c for c in self.val_catchment_names if c.startswith(base_name + '_') or c == base_name]
                
                # If iterations appear in both splits, we have an issue
                if train_iterations and val_iterations:
                    split_issues.append(f"{base_name}: {len(train_iterations)} in training, {len(val_iterations)} in validation")
            
            if split_issues:
                print(f"  WARNING: Found {len(split_issues)} catchments with iterations in both splits:")
                for issue in split_issues:
                    print(f"    - {issue}")
            else:
                print(f"  ✓ All catchment iterations are correctly kept together")
        
        # Check 3: Ensure all valid catchments are used
        all_valid_catchments = {d.base_catchment_name for d in self.datasets if d.is_valid}
        used_catchments = self.train_catchment_names.union(self.val_catchment_names)
        
        unused = all_valid_catchments - used_catchments
        
        if unused:
            print(f"  WARNING: Found {len(unused)} valid catchments not used in any split")
            print(f"    Unused: {', '.join(sorted(unused))}")
        else:
            print(f"  ✓ All valid catchments are used in the splits")
    
    def _setup_three_way_split(self, valid_datasets, split_method):
        """Split valid datasets into train, validation, and test sets"""
        if split_method != 'catchment':
            print("Warning: When test mode is enabled, using catchment-based splitting regardless of split_method parameter")
        
        # Group valid datasets by base catchment
        catchment_groups = {}
        for idx, dataset in enumerate(valid_datasets):
            base_catchment = dataset.base_catchment_name
            if base_catchment not in catchment_groups:
                catchment_groups[base_catchment] = []
            catchment_groups[base_catchment].append(idx)
        
        # Get list of unique catchments
        catchments = list(catchment_groups.keys())
        
        # Use fixed seed for reproducibility
        rng = torch.Generator().manual_seed(42)
        
        # 1. Select 10% of catchments for test set
        n_test_catchments = max(1, int(len(catchments) * 0.1))  # At least 1 catchment for test
        perm = torch.randperm(len(catchments), generator=rng)
        test_catchment_indices = perm[:n_test_catchments].tolist()
        self.test_catchment_names = {catchments[i] for i in test_catchment_indices}
        
        # 2. Collect all datasets from remaining catchments
        remaining_catchments = [c for i, c in enumerate(catchments) if i not in test_catchment_indices]
        remaining_datasets_indices = []
        for catchment in remaining_catchments:
            remaining_datasets_indices.extend(catchment_groups[catchment])
        
        # 3. Randomly split the remaining datasets into validation (20%) and training (80%)
        # This will mix iterations across catchments, making validation more similar to training
        n_remaining = len(remaining_datasets_indices)
        n_val = int(n_remaining * 0.2)
        
        # Shuffle indices and split directly
        shuffled_indices = remaining_datasets_indices.copy()
        random.Random(42).shuffle(shuffled_indices)  # Use random module for simplicity
        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]
        
        # 4. Add all test catchment datasets to test indices
        test_indices = []
        for catchment in self.test_catchment_names:
            test_indices.extend(catchment_groups[catchment])
        
        # Process the datasets based on the indices
        train_X, train_y, train_catchments = [], [], []
        val_X, val_y, val_catchments = [], [], []
        test_X, test_y, test_catchments = [], [], []
        
        # Store all catchment names for logging
        self.train_catchment_names = set()
        self.val_catchment_names = set()
        
        # Process training datasets
        for idx in train_indices:
            dataset = valid_datasets[idx]
            train_X.extend(dataset.X)
            train_y.extend(dataset.y)
            train_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
            self.train_catchment_names.add(dataset.base_catchment_name)
            
        # Process validation datasets
        for idx in val_indices:
            dataset = valid_datasets[idx]
            val_X.extend(dataset.X)
            val_y.extend(dataset.y)
            val_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
            self.val_catchment_names.add(dataset.base_catchment_name)
            
        # Process test datasets
        for idx in test_indices:
            dataset = valid_datasets[idx]
            test_X.extend(dataset.X)
            test_y.extend(dataset.y)
            test_catchments.extend([dataset.base_catchment_name] * len(dataset.X))
            
        # Print catchment information
        print(f"\nThree-way catchment-based splitting:")
        print(f"  Test catchments ({len(self.test_catchment_names)}): {', '.join(sorted(self.test_catchment_names))}")
        print(f"  Catchments in validation dataset ({len(self.val_catchment_names)}): {', '.join(sorted(self.val_catchment_names))}")
        print(f"  Catchments in training dataset ({len(self.train_catchment_names)}): {', '.join(sorted(self.train_catchment_names))}")
        
        # Convert to tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        val_X = torch.tensor(val_X, dtype=torch.float32)
        val_y = torch.tensor(val_y, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.float32)
        
        # Create datasets with catchment names
        self.train_dataset = [(x, y, c) for x, y, c in zip(train_X, train_y, train_catchments)]
        self.val_dataset = [(x, y, c) for x, y, c in zip(val_X, val_y, val_catchments)]
        self.test_dataset = [(x, y, c) for x, y, c in zip(test_X, test_y, test_catchments)]
        
        print(f"\n[SPLIT] Three-way split summary:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Training catchments: {sorted(self.train_catchment_names)}")
        print(f"  Validation catchments: {sorted(self.val_catchment_names)}")
        print(f"  Test catchments: {sorted(self.test_catchment_names)}")
        
        # Modify verification for this special case
        print("\nVerifying split integrity...")
        print(f"  ✓ Test set contains all iterations of {len(self.test_catchment_names)} complete catchments")
        print(f"  ✓ Validation and training sets contain a mix of iterations from {len(self.train_catchment_names.union(self.val_catchment_names))} remaining catchments")
        
        # Check that there's no overlap between test and other sets
        train_test_overlap = self.train_catchment_names.intersection(self.test_catchment_names)
        val_test_overlap = self.val_catchment_names.intersection(self.test_catchment_names)
        
        if train_test_overlap:
            print(f"  WARNING: Found catchments in both training and test: {', '.join(train_test_overlap)}")
        else:
            print(f"  ✓ No overlap between training and test datasets")
            
        if val_test_overlap:
            print(f"  WARNING: Found catchments in both validation and test: {', '.join(val_test_overlap)}")
        else:
            print(f"  ✓ No overlap between validation and test datasets")
            
        # Check expected overlap between training and validation
        shared_catchments = self.train_catchment_names.intersection(self.val_catchment_names)
        print(f"  ✓ Training and validation share iterations from {len(shared_catchments)} catchments (expected behavior)")
        
        # Check that all valid catchments are used
        all_valid_catchments = {d.base_catchment_name for d in self.datasets if d.is_valid}
        used_catchments = self.train_catchment_names.union(self.val_catchment_names).union(self.test_catchment_names)
        unused = all_valid_catchments - used_catchments
        
        if unused:
            print(f"  WARNING: Found {len(unused)} valid catchments not used in any split")
            print(f"    Unused: {', '.join(sorted(unused))}")
        else:
            print(f"  ✓ All valid catchments are used in the splits")

    def test_step(self, batch, batch_idx):
        """Run a test step on the test dataset"""
        x, y, catchment_names = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        
        # Track metrics for each catchment sample
        for i in range(len(catchment_names)):
            sample_mse = torch.mean((y_hat[i] - y[i])**2).item()
            
            self.test_catchment_tracker.update(
                catchment_names[i],
                sample_mse,
                y_hat[i].detach().cpu().numpy(),
                y[i].cpu().numpy()
            )
            
            self.test_prediction_performances.append({
                'mse': sample_mse,
                'prediction': y_hat[i].detach().cpu(),
                'target': y[i].cpu(),
                'catchment': catchment_names[i]
            })
        
        # Log test metrics
        metrics = {
            'test/loss': test_loss.item()
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log metrics locally
        self.local_logger.log_metrics(metrics, step=self.global_step, metric_type='test', epoch=self.current_epoch)
        
        return test_loss
    
    def on_test_epoch_end(self):
        """Log visualization of test predictions at test epoch end"""
        if len(self.test_prediction_performances) > 0:
            # Calculate KGE for each prediction
            for pred_data in self.test_prediction_performances:
                y_true = pred_data['target'].cpu().numpy()
                y_pred = pred_data['prediction'].detach().cpu().numpy()
                
                # Calculate KGE components
                r = np.corrcoef(y_true, y_pred)[0, 1]
                alpha = np.std(y_pred) / (np.std(y_true) + 1e-10)
                beta = np.mean(y_pred) / (np.mean(y_true) + 1e-10)
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                # Add KGE to prediction data
                pred_data['kge'] = kge
            
            # Sort predictions by KGE
            sorted_predictions = sorted(self.test_prediction_performances, key=lambda x: x['kge'], reverse=True)
            
            # Get total number of predictions
            n_predictions = len(sorted_predictions)
            
            # Select best, worst, and average predictions to log
            predictions_to_log = {
                'best': sorted_predictions[:3],  # Highest KGE
                'worst': sorted_predictions[-3:],  # Lowest KGE
                'average': sorted_predictions[n_predictions//2-1:n_predictions//2+2]  # Middle KGE
            }
            
            # Log selected predictions
            for category, predictions in predictions_to_log.items():
                for idx, pred_data in enumerate(predictions):
                    self._log_predictions(
                        pred_data['target'],
                        pred_data['prediction'],
                        pred_data['catchment'],
                        f'test/{category}/{pred_data["catchment"]}'
                    )
            
            # Create and log MSE distribution plot
            fig = plt.figure(figsize=(10, 5))
            mse_values = [p['mse'] for p in self.test_prediction_performances]
            plt.hist(mse_values, bins=50)
            plt.title('Distribution of Test Prediction MSE')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Count')
            self.logger.experiment.log({
                f"test_distributions/mse_distribution": wandb.Image(fig)
            })
            plt.close()
            
            # Log catchment-specific metrics
            metrics_df = self.test_catchment_tracker.get_summary()
            
            if self.logger:
                # Log individual catchment metrics
                for catchment in metrics_df.index:
                    metrics = {
                        f"test_catchments/{catchment}/mse": metrics_df.loc[catchment, 'mse'],
                        f"test_catchments/{catchment}/mae": metrics_df.loc[catchment, 'mae'],
                        f"test_catchments/{catchment}/rmse": metrics_df.loc[catchment, 'rmse'],
                        f"test_catchments/{catchment}/r2": metrics_df.loc[catchment, 'r2'],
                        f"test_catchments/{catchment}/kge": metrics_df.loc[catchment, 'kge'],
                    }
                    self.log_dict(metrics, sync_dist=True)
                    
                    # Log metrics locally
                    self.local_logger.log_metrics(
                        metrics,
                        step=None,  # Test metrics don't have steps
                        metric_type='catchment_metrics',
                        epoch=None  # Test metrics don't have epochs
                    )
                
                # Log performance comparison plot
                fig = self.test_catchment_tracker.plot_performance_comparison()
                self.logger.experiment.log({
                    f"test_catchment_metrics/comparison": wandb.Image(fig)
                })
                plt.close(fig)
                
                # Log KGE comparison plot
                kge_fig = self.test_catchment_tracker.plot_kge_comparison()
                self.logger.experiment.log({
                    f"test_catchment_metrics/kge_comparison": wandb.Image(kge_fig)
                })
                plt.close(kge_fig)
            
            # Write all collected metrics to files
            self.local_logger.on_epoch_end(None)  # Test metrics don't have epochs
            
            # Reset tracker
            self.test_prediction_performances = []
            self.test_catchment_tracker = CatchmentMetricsTracker()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=4
        )
    
    def test_dataloader(self):
        """Return a DataLoader for the test dataset if it exists"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4
        )

    def configure_callbacks(self):
        """Configure model callbacks including checkpoint saving"""
        if not self.run_id:
            raise ValueError("No WandB run ID found. Make sure WandB is properly initialized.")
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_filename = f"{self.run_id}" + "-epoch={epoch:02d}-val_loss={val/loss:.4f}"
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=checkpoint_filename,  # Now includes WandB run ID
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        # Last checkpoint name also includes run ID
        checkpoint_callback.CHECKPOINT_NAME_LAST = f"{self.run_id}-last"
        
        return [checkpoint_callback]