import pandas as pd
import os
import glob
from typing import Tuple, List, Union
import logging
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import huggingface_hub
from tkinter import filedialog
import tkinter as tk
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub

class   DataLoader:
    """
    Enhanced DataLoader that supports local files, Kaggle datasets, and remote data loading for sentiment analysis.
    Supports CSV, text files, and common benchmark datasets.
    """
    
    def __init__(self, logging_level=logging.INFO):
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    def load_local_csv(self, file_path: str = None, text_column: str = 'review', 
                      label_column: str = 'sentiment') -> Tuple[List[str], List[int]]:
        """
        Load dataset from a local CSV file, with option to browse for file.
        
        Args:
            file_path: Path to CSV file. If None, opens file browser
            text_column: Name of the column containing the text
            label_column: Name of the column containing the labels
        """
        try:
            # If no file path provided, open file browser
            if file_path is None:
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                file_path = filedialog.askopenfilename(
                    title="Select CSV file",
                    filetypes=[("CSV files", "*.csv")]
                )
                if not file_path:  # User cancelled
                    raise ValueError("No file selected")
            
            df = pd.read_csv(file_path)
            
            # Check required columns
            if not all(col in df.columns for col in [text_column, label_column]):
                raise ValueError(f"CSV must contain columns: {text_column}, {label_column}")
            
            # Convert sentiment to binary if needed
            if df[label_column].dtype == object:
                sentiment_map = {
                    'positive': 1, 'negative': 0,
                    'pos': 1, 'neg': 0,
                    '1': 1, '0': 0
                }
                df[label_column] = df[label_column].map(sentiment_map)
            
            # Remove any rows with NaN values
            df = df.dropna(subset=[text_column, label_column])
            
            self.logger.info(f"Loaded {len(df)} samples from local CSV: {file_path}")
            return df[text_column].tolist(), df[label_column].tolist()
            
        except Exception as e:
            self.logger.error(f"Error loading local CSV file: {e}")
            raise

    def load_from_kaggle(self, dataset_name: str, filename: str = None, 
                          text_column: str = 'review', label_column: str = 'sentiment',
                          use_api: bool = True) -> Tuple[List[str], List[int]]:
        """
        Load dataset from Kaggle using kagglehub library.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')
            filename: Specific file to load from the dataset. If None, uses the first CSV found.
            text_column: Name of the column containing the text data
            label_column: Name of the column containing the labels
            use_api: If True, uses Kaggle API. If False, provides manual download instructions.
        
        Returns:
            Tuple of lists containing text data and corresponding labels
        
        Raises:
            ValueError: If required columns are missing or dataset cannot be loaded
            Exception: For other unexpected errors during dataset loading
        """
        try:
            if not use_api:
                self.logger.info("Manual download instructions for Kaggle dataset:")
                print(f"1. Visit https://www.kaggle.com/datasets/{dataset_name}")
                print("2. Click 'Download' button")
                print("3. Save the file and use alternative loading method")
                return None, None

            # Download the dataset using kagglehub
            path = kagglehub.dataset_download(dataset_name)
            self.logger.info(f"Dataset downloaded to: {path}")

            # Find CSV files in the downloaded directory
            csv_files = glob.glob(os.path.join(path, '*.csv'))
            if not csv_files:
                raise ValueError("No CSV files found in the downloaded dataset")

            # Use specified file or first CSV
            target_file = filename if filename else csv_files[0]
            
            # Load the dataset
            df = pd.read_csv(target_file)
            
            # Validate required columns
            if not all(col in df.columns for col in [text_column, label_column]):
                raise ValueError(f"Required columns missing. Ensure {text_column} and {label_column} are present.")
            
            # Standardize label column
            sentiment_map = {
                'positive': 1, 'negative': 0,
                'pos': 1, 'neg': 0,
                '1': 1, '0': 0
            }
            
            # Convert labels if they are strings
            if df[label_column].dtype == object:
                df[label_column] = df[label_column].map(sentiment_map)
            
            # Remove rows with missing data
            df = df.dropna(subset=[text_column, label_column])
            
            self.logger.info(f"Loaded {len(df)} samples from Kaggle dataset: {dataset_name}")
            
            return df[text_column].tolist(), df[label_column].tolist()
        
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset: {e}")
            raise

    def load_from_huggingface(self, dataset_name: str, subset: str = None, split: str = 'train', 
                             text_column: str = 'text', label_column: str = 'label', 
                             max_samples: int = None) -> Tuple[List[str], List[int]]:
        """Existing method for loading from Hugging Face"""
        # ... (keep existing implementation)
        pass

    def load_from_url(self, url: str, text_column: str = 'review', 
                     label_column: str = 'sentiment') -> Tuple[List[str], List[int]]:
        """Existing method for loading from URL"""
        # ... (keep existing implementation)
        pass

    def prepare_dataset(self, texts: List[str], labels: List[int],
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Existing method for preparing dataset"""
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

def main():
    """
    Example usage showing different ways to load data.
    """
    from P2_sentiment_analyzer import SentimentAnalyzer
    
    loader = DataLoader()
    
    try:
        # Example 1: Loading from local CSV with file browser
        # texts, labels = loader.load_local_csv()
        
        # Example 2: Loading from local CSV with specific path
        texts, labels = loader.load_local_csv('IMDB Dataset\IMDB Dataset.csv')
        
        # Example 3: Loading from Kaggle
        # texts, labels = loader.load_from_kaggle(
        #     dataset_name='lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
        #     filename='IMDB Dataset.csv'
        # )
        
        # Example 4: Loading from Hugging Face
        # texts, labels = loader.load_from_huggingface(
        #     dataset_name='imdb',
        #     split='train',
        #     max_samples=10000
        # )
        
        # Prepare dataset
        X_train, X_test, y_train, y_test = loader.prepare_dataset(texts, labels)
        
        # Initialize and train the analyzer
        analyzer = SentimentAnalyzer(model_type='logistic')
        analyzer.train(X_train, y_train)
        
        # Evaluate
        metrics = analyzer.evaluate(X_test, y_test)
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()