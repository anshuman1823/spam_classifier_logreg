import numpy as np

def classification_metrics(y_true, y_preds):
    """
    y_true: True values
    y_preds: predicted values
    """
    y_true = np.array(y_true).ravel()
    y_preds = np.array(y_preds).ravel()

    true_pos = np.sum(np.where((y_preds == 1) & (y_true == 1), 1, 0))
    false_pos = np.sum(np.where((y_preds == 1) & (y_true == 0), 1, 0))
    true_neg = np.sum(np.where((y_preds == 0) & (y_true == 0), 1, 0))
    false_neg = np.sum(np.where((y_preds == 0) & (y_true == 1), 1, 0))
    
    recall = true_pos/(true_pos + false_neg)
    precision = true_pos/(true_pos + false_pos)
    f1_score = 2*precision*recall/(precision + recall)
    
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print(f"f1_score: {f1_score}")