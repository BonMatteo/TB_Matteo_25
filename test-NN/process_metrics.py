import os
import pandas as pd
import glob
from pathlib import Path

def get_kge_per_epoch_all_runs():
    # Get all directories in local_metrics
    base_dir = Path('local_metrics')
    directories = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # List to store all data
    all_data = []
    
    # Process each directory (each run)
    for directory in directories:
        run_name = directory.name
        
        # Find all validation metrics files
        val_files = glob.glob(str(directory / 'validation_catchment_metrics_epoch_*.csv'))
        
        for file in val_files:
            # Extract epoch number from filename
            epoch = int(file.split('_')[-1].split('.')[0])
            
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Calculate median KGE for this epoch
            median_kge = df['kge'].median()
            
            # Store data
            all_data.append({
                'run': run_name,
                'epoch': epoch,
                'median_kge': median_kge
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(all_data)
    
    # Sort by run and epoch
    result_df = result_df.sort_values(['run', 'epoch'])
    
    # Save to CSV
    result_df.to_csv('kge_per_epoch_all_runs.csv', index=False)
    print("KGE CSV file created successfully!")

def get_mse_per_epoch_all_runs():
    # Get all directories in local_metrics
    base_dir = Path('local_metrics')
    directories = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # List to store all data
    all_data = []
    
    # Process each directory (each run)
    for directory in directories:
        run_name = directory.name
        
        # Find all validation metrics files
        val_files = glob.glob(str(directory / 'validation_catchment_metrics_epoch_*.csv'))
        
        for file in val_files:
            # Extract epoch number from filename
            epoch = int(file.split('_')[-1].split('.')[0])
            
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Calculate mean MSE for this epoch
            mean_mse = df['mse'].mean()
            
            # Store data
            all_data.append({
                'run': run_name,
                'epoch': epoch,
                'mean_mse': mean_mse
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(all_data)
    
    # Sort by run and epoch
    result_df = result_df.sort_values(['run', 'epoch'])
    
    # Save to CSV
    result_df.to_csv('mse_per_epoch_all_runs.csv', index=False)
    print("MSE CSV file created successfully!")

def get_test_mse_all_runs():
    # Get all directories in local_metrics
    base_dir = Path('local_metrics')
    directories = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # List to store all data
    all_data = []
    
    # Process each directory (each run)
    for directory in directories:
        run_name = directory.name
        
        # Find test metrics file
        test_file = directory / 'test_catchment_metrics_epoch_None.csv'
        
        if test_file.exists():
            # Read the CSV file
            df = pd.read_csv(test_file)
            
            # Calculate mean MSE for test phase
            mean_mse = df['mse'].mean()
            
            # Store data
            all_data.append({
                'run': run_name,
                'mean_mse': mean_mse
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(all_data)
    
    # Sort by run name
    result_df = result_df.sort_values('run')
    
    # Save to CSV
    result_df.to_csv('test_mse_all_runs.csv', index=False)
    print("Test MSE CSV file created successfully!")

if __name__ == "__main__":
    get_kge_per_epoch_all_runs()
    get_mse_per_epoch_all_runs()
    get_test_mse_all_runs() 