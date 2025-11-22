"""Basic model evaluation helpers."""
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    res = {
        "accuracy": accuracy_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(y_test, preds, digits=4),
    }
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            res["roc_auc"] = roc_auc_score(y_test, probs)
    except Exception:
        pass

    return res
