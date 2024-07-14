from sklearn.metrics import roc_auc_score, f1_score
# variables are expected to be 1D
def calculate_auc(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return roc_auc_score(y_true, y_pred)
def calculate_f1(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return f1_score(y_true, y_pred, average = 'macro')