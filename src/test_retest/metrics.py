
def specificity_score(y_true, y_pred):
    """
    Compute true negative rate.
    TN / (TN + FP)
    """
    TN = 0
    FP = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == 0 and y_p == 0:
            TN += 1
        if y_t == 0 and y_p == 1:
            FP += 1
    
    return TN / (TN + FP)
