from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score


def classify_by_threshold(diff_list, threshold):
    y_pred = [1 if diff > threshold else 0 for diff in diff_list]
    return y_pred


def evaluate_model(diff_list, threshold):
    """
    Evaluate model performance based on the given threshold.

    Returns
    -------
    A dictionary containing 
        precision, 
        recall, 
        roc_auc, 
        accuracy.
    """
    y_true, y_pred = classify_by_threshold(diff_list, threshold)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "accuracy": acc,
    }


def evaluate_over_thresholds(diff_list, thresholds):
    """
    Evaluate model performance over a range of thresholds.

    Returns
    -------
    A list of dictionaries containing evaluation metrics for each threshold.
    """
    results = []
    for t in thresholds:
        metrics = evaluate_model(diff_list, t)
        metrics['threshold'] = t
        results.append(metrics)
    return results