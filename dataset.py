import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class HMDataset(Dataset):
    """Memory-efficient H&M dataset loader with temporal features"""

    def __init__(self, data_path: str, split: str = 'train', seq_len: int = 28, pred_len: int = 7):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train' or 'test'
            seq_len: Input sequence length in days
            pred_len: Prediction horizon length
        """
        self.transactions = pd.read_csv(f'{data_path}/transactions_train.csv', parse_dates=['t_dat'])
        self.articles = pd.read_csv(f'{data_path}/articles.csv')
        self.customers = pd.read_csv(f'{data_path}/customers.csv')
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Merge datasets
        self._preprocess()
        self._create_temporal_features()

    def _preprocess(self):
        """Merge datasets and handle missing values"""
        # Merge transactions with articles
        self.df = self.transactions.merge(self.articles, on='article_id', how='left')

        # Merge with customers
        self.df = self.df.merge(self.customers, on='customer_id', how='left')

        # Handle missing values
        self.df['age'].fillna(self.df['age'].median(), inplace=True)
        self.df['club_member_status'].fillna('UNKNOWN', inplace=True)

    def _create_temporal_features(self):
        """Create time-based features"""
        self.df['day_of_week'] = self.df['t_dat'].dt.dayofweek
        self.df['month'] = self.df['t_dat'].dt.month
        self.df['year'] = self.df['t_dat'].dt.year
        self.df['is_weekend'] = self.df['day_of_week'].isin([5,6]).astype(int)

    def __len__(self) -> int:
        return len(self.df) - self.seq_len - self.pred_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (seq_len, n_features), (pred_len,)"""
        sequence = self.df.iloc[idx:idx+self.seq_len]
        target = self.df.iloc[idx+self.seq_len:idx+self.seq_len+self.pred_len]['price']

        # Convert to tensors
        features = torch.FloatTensor(sequence[['price', 'day_of_week', 'month', 'age']].values)
        target = torch.FloatTensor(target.values)

        return features, target
