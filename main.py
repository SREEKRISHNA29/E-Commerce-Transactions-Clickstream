from src.data_preprocessing import load_and_preprocess
from src.train_model import train_models
from src.evaluate_model import evaluate_models

def main():
    clickstream_path = "data/clickstream.csv"
    transactions_path = "data/transactions.csv"

    X_train, X_test, y_train, y_test = load_and_preprocess(clickstream_path, transactions_path)
    train_models(X_train, y_train)
    evaluate_models(X_test, y_test)

if __name__ == "__main__":
    main()
