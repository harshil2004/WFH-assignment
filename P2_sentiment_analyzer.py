import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import joblib
from datetime import datetime

class SentimentAnalyzer:
    """
    Enhanced sentiment analyzer with improved preprocessing, model management,
    and integration capabilities.
    """
    
    def __init__(self, 
                 model_type: str = 'logistic',
                 vectorizer_params: Optional[Dict] = None,
                 model_params: Optional[Dict] = None):
        """
        Initialize the sentiment analyzer with customizable parameters.
        
        Args:
            model_type: Type of model ('logistic', 'naive_bayes', or 'svm')
            vectorizer_params: Custom parameters for TF-IDF vectorizer
            model_params: Custom parameters for the selected model
        """
        # Set up logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK resources
        try:
            for resource in ['punkt', 'stopwords', 'wordnet']:
                nltk.download(resource, quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.error(f"Error downloading NLTK resources: {e}")
            raise
        
        # Default vectorizer parameters
        default_vectorizer_params = {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 3),
            'strip_accents': 'unicode',
            'norm': 'l2'
        }
        
        # Update with custom parameters if provided
        if vectorizer_params:
            default_vectorizer_params.update(vectorizer_params)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(**default_vectorizer_params)
        
        # Initialize models with default parameters
        default_model_params = {
            'logistic': {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'},
            'naive_bayes': {'alpha': 1.0},
            'svm': {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'}
        }
        
        # Update with custom parameters if provided
        if model_params:
            for model_name in default_model_params:
                if model_name in model_params:
                    default_model_params[model_name].update(model_params[model_name])
        
        # Initialize model dictionary
        self.models = {
            'logistic': LogisticRegression(**default_model_params['logistic']),
            'naive_bayes': MultinomialNB(**default_model_params['naive_bayes']),
            'svm': LinearSVC(**default_model_params['svm'])
        }
        
        # Set model type
        if model_type not in self.models:
            raise ValueError(f"Model type must be one of {list(self.models.keys())}")
        self.model_type = model_type
        self.model = self.models[model_type]
        
        # Initialize state variables
        self.is_fitted = False
        self.feature_names = None
        self.training_size = 0
        self.class_distribution = None

    def preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing with additional cleaning steps.
        """
        try:
            # Convert to string if not already
            text = str(text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters and numbers but keep important punctuation
            text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short words
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
            
            # Rejoin tokens
            return ' '.join(tokens)
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {e}")
            raise

    def prepare_data(self, texts: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data with enhanced error checking and logging.
        """
        try:
            # Input validation
            if not texts or not labels:
                raise ValueError("Empty input data")
            if len(texts) != len(labels):
                raise ValueError("Number of texts and labels must match")
            
            # Preprocess all texts
            self.logger.info("Starting text preprocessing...")
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Vectorize texts
            self.logger.info("Vectorizing texts...")
            if not self.is_fitted:
                features = self.vectorizer.fit_transform(processed_texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
            else:
                features = self.vectorizer.transform(processed_texts)
            
            # Store class distribution
            unique, counts = np.unique(labels, return_counts=True)
            self.class_distribution = dict(zip(unique, counts))
            
            self.logger.info(f"Data preparation completed. Feature matrix shape: {features.shape}")
            return features, np.array(labels)
        except Exception as e:
            self.logger.error(f"Error in data preparation: {e}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced training with detailed metrics and parameter tuning.
        """
        try:
            self.logger.info("Starting model training...")
            self.training_size = X_train.shape[0]
            
            # Define parameter grid for each model
            param_grids = {
                'logistic': {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced'],
                    'max_iter': [1000]
                },
                'naive_bayes': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                },
                'svm': {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced'],
                    'max_iter': [1000]
                }
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model,
                param_grids[self.model_type],
                cv=5,
                scoring=['accuracy', 'precision', 'recall', 'f1'],
                refit='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.is_fitted = True
            
            # Collect training metrics
            training_metrics = {
                'best_params': grid_search.best_params_,
                'cv_results': {
                    'mean_accuracy': grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_],
                    'mean_precision': grid_search.cv_results_['mean_test_precision'][grid_search.best_index_],
                    'mean_recall': grid_search.cv_results_['mean_test_recall'][grid_search.best_index_],
                    'mean_f1': grid_search.cv_results_['mean_test_f1'][grid_search.best_index_]
                }
            }
            
            self.logger.info("Model training completed successfully")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced evaluation with detailed metrics and error analysis.
        """
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate detailed metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Add error analysis
            error_indices = np.where(y_test != y_pred)[0]
            metrics['error_analysis'] = {
                'num_errors': len(error_indices),
                'error_rate': len(error_indices) / len(y_test)
            }
            
            self.logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            raise

    def predict(self, texts: List[str], return_proba: bool = False) -> np.ndarray:
        """
        Make predictions with optional probability scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare data
            features, _ = self.prepare_data(texts, [0] * len(texts))
            
            # Make predictions
            predictions = self.model.predict(features)
            
            # Get probability scores if requested and available
            if return_proba and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
                return predictions, probabilities
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise

    def save_model(self, path: str) -> None:
        """
        Save the trained model and vectorizer.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create a dictionary with all necessary components
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_size': self.training_size,
                'class_distribution': self.class_distribution,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a previously saved model.
        """
        try:
            # Load model data
            model_data = joblib.load(path)
            
            # Restore all components
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_size = model_data['training_size']
            self.class_distribution = model_data['class_distribution']
            self.is_fitted = True
            
            self.logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        """
        return {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names) if self.feature_names is not None else 0,
            'training_size': self.training_size,
            'class_distribution': self.class_distribution,
            'vectorizer_params': self.vectorizer.get_params(),
            'model_params': self.model.get_params()
        }

def main():
    """
    Example usage with the data loader.
    """
    from P2_data_loader import DataLoader
    
    try:
        # Initialize data loader and analyzer
        loader = DataLoader()
        analyzer = SentimentAnalyzer(model_type='logistic')
        
        # Load sample data
        texts = [
            "This product is amazing! I love everything about it.",
            "Terrible experience, would not recommend.",
            "Good quality for the price, happy with my purchase.",
            "Waste of money, very disappointed.",
            # Add more examples here...
        ]
        labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative
        
        # Prepare dataset
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Prepare and train
        X_train_prepared, y_train_prepared = analyzer.prepare_data(X_train, y_train)
        training_metrics = analyzer.train(X_train_prepared, y_train_prepared)
        
        # Evaluate
        X_test_prepared, y_test_prepared = analyzer.prepare_data(X_test, y_test)
        evaluation_metrics = analyzer.evaluate(X_test_prepared, y_test_prepared)
        
        # Print results
        print("\nTraining Metrics:")
        print(training_metrics)
        print("\nEvaluation Metrics:")
        print(evaluation_metrics)
        
        # Save model
        analyzer.save_model('sentiment_model.joblib')
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()