import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime

class BaseflowDataset(Dataset):
    """Dataset for baseflow prediction with preprocessing and caching.
    
    Features:
    - Loads and processes baseflow data from CSV files
    - Filters years based on Baseflow Index (BFI) criteria
    - Creates input/output sequences for each valid year
    - Caches processed data for faster subsequent access
    - Tracks invalid catchments and removed years
    """
    def __init__(self, file_path, years_range=(1994, 2022), cache_dir='preprocessed_data', full_data=False):
        # Extract both full and base catchment names
        filename = os.path.basename(file_path)
        self.full_catchment_name = filename[len('baseflow_'):-len('.csv')]  # e.g., "Allenbach_0"
        
        # Extract base catchment name (everything before the last underscore)
        self.base_catchment_name = self.full_catchment_name.rsplit('_', 1)[0]  # e.g., "Allenbach"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache file path with full_data flag in filename
        cache_suffix = '_full' if full_data else ''
        cache_file = os.path.join(
            cache_dir, 
            f"preprocessed_{filename.replace('.csv', '')}{cache_suffix}_data.csv"
        )
        
        # Files to store removed years and invalid catchments info
        self.removed_years_file = os.path.join(cache_dir, '00_removed_years_BFI.csv')
        self.invalid_catchments_file = os.path.join(cache_dir, '00_invalid_catchments.csv')
        
        try:
            self._load_or_process_data(file_path, cache_file, years_range, full_data)
            self.is_valid = True
        except ValueError as e:
            self._log_invalid_catchment(str(e))
            self.is_valid = False
            # Initialize empty arrays to prevent errors
            self.X = np.array([])
            self.y = np.array([])

    def _log_invalid_catchment(self, reason):
        """Log details of invalid catchments to tracking file"""
        invalid_catchment_info = {
            'catchment': self.full_catchment_name,
            'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': reason
        }
        
        # Check if file exists and read existing data
        if os.path.exists(self.invalid_catchments_file):
            existing_df = pd.read_csv(self.invalid_catchments_file)
            
            # Check if this catchment with this reason already exists
            if ((existing_df['catchment'] == self.full_catchment_name) & 
                (existing_df['reason'] == reason)).any():
                # Entry already exists, don't add duplicate
                return
            
            # Add new entry to existing data
            new_df = pd.DataFrame([invalid_catchment_info])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(self.invalid_catchments_file, index=False)
        else:
            # Create new file
            df = pd.DataFrame([invalid_catchment_info])
            df.to_csv(self.invalid_catchments_file, index=False)

    def _load_or_process_data(self, file_path, cache_file, years_range, full_data):
        """Load from cache if available, otherwise process raw data"""
        if os.path.exists(cache_file):
            self._load_from_cache(cache_file)
        else:
            self._process_raw_data(file_path, cache_file, years_range, full_data)

    def _load_from_cache(self, cache_file):
        """Load preprocessed data from cache file"""
        df = pd.read_csv(cache_file)
        X_cols = [col for col in df.columns if col.startswith('X_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]
        
        if len(X_cols) == 0 or len(y_cols) == 0:
            raise ValueError("Cache file is corrupted or empty")
        
        self.X = df[X_cols].values.astype(np.float32)
        self.y = df[y_cols].values.astype(np.float32)

    def _process_raw_data(self, file_path, cache_file, years_range, full_data):
        """Process raw CSV data and save to cache"""
        # Read and preprocess the data
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Compute hydrological year
        df['hydro_year'] = df['time'].apply(
            lambda d: d.year + 1 if d.month >= 10 else d.year
        )
        df['base_date'] = df.apply(
            lambda row: pd.Timestamp(row['hydro_year'] - 1, 10, 1),
            axis=1
        )
        df['hydro_day_of_year'] = (df['time'] - df['base_date']).dt.days + 1
        
        df['year'] = df['hydro_year']
        df['day_of_year'] = df['hydro_day_of_year']
        
        # Filter by years range
        df = df[(df['year'] >= years_range[0]) & (df['year'] <= years_range[1])]
        
        # Calculate BFI and filter years
        valid_years, removed_years = self._calculate_and_filter_bfi(df)
        
        if len(valid_years) == 0:
            raise ValueError("No valid years remaining after BFI filtering")
        
        # Save removed years information
        self._save_removed_years(removed_years)
        
        # Filter dataset to valid years only
        df = df[df['year'].isin(valid_years)]
        
        # Process data into sequences
        X_list, y_list = self._create_sequences(df, valid_years, full_data)
        
        if not X_list or not y_list:
            raise ValueError("No valid sequences created after processing")
        
        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)
        
        # Save to cache
        self._save_to_cache(cache_file)

    def _calculate_and_filter_bfi(self, df):
        """Filter years based on Baseflow Index criteria (0.1 <= BFI <= 0.9)
        
        BFI is calculated as the ratio of total baseflow to total streamflow.
        Years with invalid BFI values are tracked for logging.
        """
        valid_years = []
        removed_years = []
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            total_streamflow = year_data['Q'].sum()
            total_baseflow = year_data['baseflow'].sum()
            
            if total_streamflow > 0:
                bfi = total_baseflow / total_streamflow
                if 0.1 <= bfi <= 0.9:
                    valid_years.append(year)
                else:
                    removed_years.append({
                        'catchment': self.full_catchment_name,
                        'year': year,
                        'BFI': bfi,
                        'total_streamflow': total_streamflow,
                        'total_baseflow': total_baseflow
                    })
        
        return valid_years, removed_years

    def _save_removed_years(self, removed_years):
        """Save information about years removed due to BFI criteria"""
        if not removed_years:
            return
        
        new_df = pd.DataFrame(removed_years)
        
        # Check if file exists and read existing data
        if os.path.exists(self.removed_years_file):
            existing_df = pd.read_csv(self.removed_years_file)
            
            # Filter out entries that already exist
            merged_df = pd.merge(
                new_df, existing_df, 
                on=['catchment', 'year'], 
                how='left', 
                indicator=True
            )
            
            # Only keep entries that don't exist in the file
            unique_entries = merged_df[merged_df['_merge'] == 'left_only']
            if '_merge' in unique_entries.columns:
                unique_entries = unique_entries.drop(columns=['_merge'])
            
            if not unique_entries.empty:
                # Add only unique entries to the file
                updated_df = pd.concat([existing_df, unique_entries], ignore_index=True)
                updated_df.to_csv(self.removed_years_file, index=False)
        else:
            # Create new file
            new_df.to_csv(self.removed_years_file, index=False)

    def _create_sequences(self, df, valid_years, full_data):
        """Create input and output sequences for each valid year
        
        For each valid year, creates:
        - Input (X): daily Q (streamflow) values, with Pmean and Tmean if full_data=True
        - Output (y): daily baseflow values
        Each sequence covers 366 days (to accommodate leap years)
        """
        X_list = []
        y_list = []
        
        for year in valid_years:
            year_data = df[df['year'] == year]
            
            Q_sequence = np.zeros(366)
            Pmean_sequence = np.zeros(366) if full_data else None
            Tmean_sequence = np.zeros(366) if full_data else None
            baseflow_sequence = np.zeros(366)
            
            for _, row in year_data.iterrows():
                day_idx = int(row['day_of_year']) - 1
                if 0 <= day_idx < 366:
                    Q_sequence[day_idx] = row['Q']
                    if full_data:
                        Pmean_sequence[day_idx] = row['Pmean']
                        Tmean_sequence[day_idx] = row['Tmean']
                    baseflow_sequence[day_idx] = row['baseflow']
            
            if full_data:
                X_sequence = np.concatenate([Q_sequence, Pmean_sequence, Tmean_sequence])
            else:
                X_sequence = Q_sequence
            
            X_list.append(X_sequence)
            y_list.append(baseflow_sequence)
        
        return X_list, y_list

    def _save_to_cache(self, cache_file):
        """Save processed X and y data to cache file for faster future loading"""
        X_cols = [f'X_{i}' for i in range(self.X.shape[1])]
        y_cols = [f'y_{i}' for i in range(self.y.shape[1])]
        
        preprocessed_df = pd.DataFrame(
            np.concatenate([self.X, self.y], axis=1),
            columns=X_cols + y_cols
        )
        
        preprocessed_df.to_csv(cache_file, index=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Use base_catchment_name for model evaluation/graphs
        return self.X[idx], self.y[idx], self.base_catchment_name

def load_catchment_datasets(file_paths, years_range=(1994, 2022), cache_dir='preprocessed_data', full_data=False):
    """Load multiple catchment datasets, handling invalid ones gracefully"""
    datasets = []
    skipped_catchments = []
    
    for file_path in tqdm(file_paths, desc="Loading catchments"):
        dataset = BaseflowDataset(file_path, years_range, cache_dir, full_data)
        if dataset.is_valid:
            datasets.append(dataset)
        else:
            # Use full catchment name for logging skipped catchments
            filename = os.path.basename(file_path)
            full_catchment_name = filename[len('baseflow_'):-len('.csv')]
            skipped_catchments.append(full_catchment_name)
    
    if not datasets:
        raise ValueError("No valid catchments found")
    
    return datasets

class BaseflowDataProcessor:
    def prepare_data(self, X, y, batch_size=32, train_split=0.8):
        # Convert to tensors without altering the data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Split data
        train_size = int(len(X) * train_split)
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader

    def inverse_transform_predictions(self, predictions):
        # No transformation needed since data was not scaled
        return predictions 

class CatchmentMetricsTracker:
    """Tracks and analyzes model performance by catchment"""
    def __init__(self):
        self.metrics = {}
        
    def update(self, catchment_name, loss, predictions, targets):
        # catchment_name will be the base_catchment_name from __getitem__
        if catchment_name not in self.metrics:
            self.metrics[catchment_name] = {
                'losses': [],
                'predictions': [],
                'targets': []
            }
        
        self.metrics[catchment_name]['losses'].append(loss)
        self.metrics[catchment_name]['predictions'].append(predictions)
        self.metrics[catchment_name]['targets'].append(targets)
    
    def compute_metrics(self):
        """Computes aggregate metrics for each catchment"""
        results = {}
        for catchment, data in self.metrics.items():
            # Now working with base catchment names
            losses = np.array(data['losses'])
            predictions = np.concatenate(data['predictions'])
            targets = np.concatenate(data['targets'])
            
            # Calculate metrics
            mse = np.mean(losses)
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(mse)
            
            # Calculate R² score
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            results[catchment] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        
        return results
    
    def get_summary(self):
        """Returns a pandas DataFrame with all metrics"""
        metrics = self.compute_metrics()
        return pd.DataFrame.from_dict(metrics, orient='index')
    
    def plot_performance_comparison(self):
        """Creates visualization of performance across catchments"""
        metrics_df = self.get_summary()
        
        plt.figure(figsize=(15, 10))
        
        # Plot MSE comparison
        plt.subplot(2, 2, 1)
        metrics_df['mse'].plot(kind='bar')
        plt.title('MSE by Catchment')
        plt.xticks(rotation=45)
        
        # Plot R² comparison
        plt.subplot(2, 2, 2)
        metrics_df['r2'].plot(kind='bar')
        plt.title('R² Score by Catchment')
        plt.xticks(rotation=45)
        
        # Plot RMSE comparison
        plt.subplot(2, 2, 3)
        metrics_df['rmse'].plot(kind='bar')
        plt.title('RMSE by Catchment')
        plt.xticks(rotation=45)
        
        # Plot MAE comparison
        plt.subplot(2, 2, 4)
        metrics_df['mae'].plot(kind='bar')
        plt.title('MAE by Catchment')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt.gcf() 