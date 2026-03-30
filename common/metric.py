import numpy as np

def get_confusion_matrix(y_pred, y_true, n_classes):
    """
    Calculates confusion matrix
    :param y_pred: the predicted labels
    :param y_true: the ground truth labels
    :param n_classes: the number of classes
    :return: confusion matrix
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    mask = (y_true_flat >= 0) & (y_pred_flat < n_classes)
    label = n_classes * y_true_flat[mask].astype('int') + y_pred_flat.astype('int')
    count = np.bincount(label, minlength=n_classes**2)
    return np.reshape(count, (n_classes, n_classes))


def get_iou(cm):
    """
    Calculates IoU metric
    :param cm: confusion matrix
    :return: class-wise IoU metric
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    iou = np.divide(
        tp, union,
        out=np.full_like(tp, np.nan, dtype=np.float32),
        where=union != 0)
    return iou


def get_dice(cm):
    """
    Calculates dice metric
    :param cm: confusion matrix
    :return: class-wise dice metric
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = 2*tp + fp + fn
    dice = np.divide(2*tp, union,
                     out=np.full_like(tp, np.nan, dtype=np.float32),
                     where=union != 0)
    return dice