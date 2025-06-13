import pytorch_lightning as pl
from model import BaseflowNN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
import argparse
from datetime import datetime
import os
import json

class PhaseCheckpoint(pl.Callback):
    """Callback that saves checkpoints at the end of learning rate phases"""
    def __init__(self, warmup_epochs, constant_epochs, decay_epochs):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.constant_epochs = constant_epochs
        self.decay_epochs = decay_epochs
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Checkpoint at end of constant phase
        if current_epoch == (self.warmup_epochs + self.constant_epochs):
            trainer.save_checkpoint(os.path.join(self.checkpoint_dir, "end_of_constant_phase.ckpt"))
        
        # Checkpoint at end of decay phase
        elif current_epoch == (self.warmup_epochs + self.constant_epochs + self.decay_epochs):
            trainer.save_checkpoint(os.path.join(self.checkpoint_dir, "end_of_training.ckpt"))
            trainer.should_stop = True

class BaseflowTrainer:
    """Basic trainer for the baseflow neural network without PyTorch Lightning
    
    Provides traditional training loop implementation with manual epoch tracking,
    validation, learning rate scheduling, and early stopping.
    """
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def train_epoch(self, train_loader):
        """Run a single training epoch
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Run validation on the current model state
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """Train the model with early stopping
        
        Args:
            train_loader: DataLoader with training data
            val_loader: DataLoader with validation data
            epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if validation doesn't improve after this many epochs
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

def main():
    """Main training execution function
    
    Handles:
    1. Command line argument parsing
    2. WandB configuration and logging setup
    3. Model initialization with configured parameters
    4. Training execution with PyTorch Lightning
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train baseflow prediction model')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='Custom name for this run')
    parser.add_argument('--split_method', type=str, default='random',
                        choices=['random', 'catchment'],
                        help='Method to split training and validation data')
    parser.add_argument('--test', action='store_true', 
                        help='Enable test mode (10%% test, 20%% validation, 70%% training)')
    args = parser.parse_args()
    
    # Create run name with timestamp and optional custom name
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        display_name = f"{current_date}-{args.run_name}"
        run_id = f"run-{current_date}-{args.run_name}"
    else:
        display_name = f"{current_date}-baseflow-dnn"
        run_id = f"run-{current_date}-baseflow-dnn"
    
    # Default configuration
    config = {
        'layer_dimensions': [4096],     # Size of network layers
        'dropout_rate': 0.1,            # Dropout for regularization, 0.1 normal -> 0.3-0.5 if overfitting
        'full_data': True,             # Whether to use all features (Q, Pmean, Tmean) or just Q
        'num_blocks': 5,                # Number of residual blocks
        'layers_per_block': 4,          # Layers in each block
        'batch_size': 128,              # Samples per batch
        'years_range': (1994, 2022),    # Year range to use for training
        'input_features': ['Q', 'Pmean', 'Tmean'],  # Available input features
        'target_variable': 'baseflow',   # Target variable to predict
        'split_method': args.split_method,  # Method for splitting train/validation data
        'test': args.test,               # Whether to use test mode (three-way split)
        'init_lr': 1e-15,     # Starting learning rate (very small)
        'peak_lr': 1e-4,      # Maximum learning rate during training
        'final_lr': 1e-6,     # Final learning rate after decay
        'warmup_epochs': 3,   # Epochs for warmup phase
        'constant_epochs': 5, # Epochs at peak learning rate
        'decay_epochs': 2     # Epochs for learning rate decay
    }
    
    # Load configuration from environment variable if available
    if "MODEL_CONFIG" in os.environ:
        config_path = os.environ["MODEL_CONFIG"]
        try:
            with open(config_path, 'r') as f:
                env_config = json.load(f)
                # Update config with values from environment
                config.update(env_config)
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_path}: {e}")
            print("Using default configuration")
    
    # Initialize WandB for experiment tracking
    wandb_logger = WandbLogger(
        project='baseflow-prediction',
        name=display_name,
        id=run_id,
        config=config
    )
    
    # Define metric tracking configuration
    wandb_logger.experiment.define_metric(
        "train/loss",
        step_metric="step",
        summary="min"
    )
    
    wandb_logger.experiment.define_metric(
        "train/learning_rate",
        step_metric="step"
    )
    
    wandb_logger.experiment.define_metric(
        "val/loss",
        step_metric="step",
        summary="min"
    )
    
    if args.test:
        wandb_logger.experiment.define_metric(
            "test/loss",
            step_metric="step",
            summary="min"
        )
    
    # Group metrics for better visualization in WandB
    metric_groups = {
        "training": {
            "metrics": [
                "train/loss",
                "train/learning_rate"
            ],
            "x_axis": "step"
        },
        "validation": {
            "metrics": [
                "val/loss"
            ],
            "x_axis": "step"
        },
        "catchment_metrics": {
            "metrics": [
                "train_catchment_metrics/*", 
                "val_catchment_metrics/*"
            ],
            "x_axis": "epoch"
        },
        "kge_metrics": {
            "metrics": [
                "training_catchments/*/kge",
                "validation_catchments/*/kge"
            ],
            "x_axis": "epoch"
        }
    }
    
    if args.test:
        metric_groups["testing"] = {
            "metrics": [
                "test/loss"
            ],
            "x_axis": "step"
        }
        metric_groups["catchment_metrics"]["metrics"].append("test_catchment_metrics/*")
        metric_groups["kge_metrics"]["metrics"].append("test_catchments/*/kge")
    
    # Update WandB configuration
    wandb_logger.experiment.config.update({
        "metric_groups": metric_groups,
        "axis_config": {
            "Loss": {
                "name": "Mean Squared Error",
                "direction": "minimize"
            },
            "Learning Rate": {
                "name": "Learning Rate Value",
                "direction": "auto"
            }
        }
    })
    
    # Initialize model with configuration
    model = BaseflowNN(
        layer_dimensions=config['layer_dimensions'],
        dropout_rate=config['dropout_rate'],
        init_lr=config['init_lr'],
        peak_lr=config['peak_lr'],
        final_lr=config['final_lr'],
        warmup_epochs=config['warmup_epochs'],
        constant_epochs=config['constant_epochs'],
        decay_epochs=config['decay_epochs'],
        full_data=config['full_data'],
        batch_size=config['batch_size'],
        num_blocks=config['num_blocks'],
        layers_per_block=config['layers_per_block'],
        split_method=config['split_method'],
        test=config['test']
    )
    
    # Initialize PyTorch Lightning trainer
    phase_checkpoint = PhaseCheckpoint(
        warmup_epochs=config['warmup_epochs'],
        constant_epochs=config['constant_epochs'],
        decay_epochs=config['decay_epochs']
    )
    
    trainer = pl.Trainer(
        max_epochs=config['warmup_epochs'] + config['constant_epochs'] + config['decay_epochs'],
        accelerator='auto',
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        val_check_interval=1.0,
        callbacks=[phase_checkpoint]
    )
    
    # Train the model
    trainer.fit(model)
    
    # Test the model if in test mode
    if args.test:
        print("\nRunning test evaluation...")
        trainer.test(model)

if __name__ == "__main__":
    main() 