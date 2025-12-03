import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(clickstream_path, transactions_path):
    # Load datasets
    clicks = pd.read_csv(clickstream_path)
    transactions = pd.read_csv(transactions_path)
    
    # Merge both datasets on user/session ID
    df = pd.merge(clicks, transactions, on="user_id", how="left")

    # Create target variable: purchase = 1 if transaction exists else 0
    df["purchase"] = df["transaction_id"].notnull().astype(int)
    
    # Feature Engineering
    df["total_clicks"] = df.groupby("session_id")["event_type"].transform("count")
    df["unique_products_viewed"] = df.groupby("session_id")["product_id"].transform("nunique")
    df["cart_events"] = (df["event_type"] == "cart").astype(int)
    
    # Drop unneeded columns
    df = df.drop(["transaction_id", "timestamp"], axis=1, errors="ignore")

    # Handle missing values
    df = df.fillna("Unknown")

    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Split features and labels
    X = df.drop("purchase", axis=1)
    y = df["purchase"]

    # Standardize numerical columns
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
