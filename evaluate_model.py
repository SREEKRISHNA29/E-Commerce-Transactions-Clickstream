from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

def evaluate_models(X_test, y_test):
    model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
    for name in model_names:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)

        print(f"\n=== {name} Evaluation ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
