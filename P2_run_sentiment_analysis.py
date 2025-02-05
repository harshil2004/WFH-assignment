from P2_data_loader import DataLoader
from P2_sentiment_analyzer import SentimentAnalyzer
import logging

def train_model(loader, analyzer, texts, labels):
    """
    Train the sentiment analysis model.

    Args:
    - loader: DataLoader instance
    - analyzer: SentimentAnalyzer instance
    - texts: list of review texts
    - labels: list of sentiment labels
    """
    # Prepare dataset
    X_train, X_test, y_train, y_test = loader.prepare_dataset(texts, labels)
    
    # Train the model
    X_train_prepared, y_train_prepared = analyzer.prepare_data(X_train, y_train)
    analyzer.train(X_train_prepared, y_train_prepared)

    return X_test, y_test, analyzer

def test_model(analyzer, X_test, y_test):
    """
    Test the sentiment analysis model.

    Args:
    - analyzer: SentimentAnalyzer instance
    - X_test: test features
    - y_test: true sentiment labels for testing
    """
    # Prepare test data
    X_test_prepared, y_test_prepared = analyzer.prepare_data(X_test, y_test)
    
    # Evaluate model
    metrics = analyzer.evaluate(X_test_prepared, y_test_prepared)
    
    return metrics

def main():
    # Configure logging for better error tracking
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize loader and analyzer
        loader = DataLoader()
        analyzer = SentimentAnalyzer(model_type='logistic')
        
        # Load dataset (choose one method)
        # Method 1: From CSV
        texts, labels = loader.load_local_csv('IMDB Dataset\IMDB Dataset.csv')
        
        # Train model once
        X_test, y_test, analyzer = train_model(loader, analyzer, texts, labels)
        
        # Test model
        metrics = test_model(analyzer, X_test, y_test)
        
        # Enhanced metrics printing with robust error handling
        print("\nFinal Evaluation Metrics:")
        for metric, value in metrics.items():
            try:
                # Check if value is a number before formatting
                if value is not None and isinstance(value, (int, float)):
                    print(f"{metric.capitalize()}: {value:.3f}")
                else:
                    print(f"{metric.capitalize()}: N/A or Invalid")
            except Exception as metric_error:
                logger.error(f"Error processing metric {metric}: {metric_error}")
                print(f"{metric.capitalize()}: Error in calculation")

    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {e}")
        print("Sentiment analysis failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
