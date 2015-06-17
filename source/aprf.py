import numpy as np


def calc_aprf(gold, prediction, f_beta_sq=1.0):
    """Compute the accuracy, precision, recall and F_beta^2.
    :param gold -- the gold standard annotations
    :param prediction -- the system predictions
    :param f_beta_sq -- the beta parameter for F_beta
    """
    tp, fp, fn, tn = calc_confusion_matrix(gold, prediction)

    accuracy = 1.0 * (tp + tn) / (tp + fp + fn + tn)
    if tp > 0:
        precision = 1.0 * tp / (tp + fp)
        recall = 1.0 * tp / (tp + fn)
        f_beta = (1.0 + f_beta_sq) * precision * recall / (f_beta_sq * precision + recall)
    else:
        precision = 1.0
        recall = 0.0
        f_beta = 0.0
    return accuracy, precision, recall, f_beta


def calc_confusion_matrix(gold, prediction):
    """Compute the confusion matrix.
    :param gold -- the gold standard annotations
    :param prediction -- the system predictions
    """
    yy = sum(gold * prediction)
    ny = sum((1-gold) * prediction)
    yn = sum(gold * (1-prediction))
    nn = sum((1-gold) * (1-prediction))
    return yy, ny, yn, nn


def multi_calc_aprf(gold, prediction, f_beta_sq=1.0):
    """Compute the accuracy, precision, recall and F_beta^2 of multiple models at once.
    :param gold -- the gold standard annotations of each model.
    :param prediction -- the system predictions in each model.
    :param f_beta_sq -- the beta parameter for F_beta.
    """
    tp, fp, fn, tn = multi_calc_confusion_matrix(gold, prediction)

    accuracy = 1.0 * (tp + tn) / (tp + fp + fn + tn)
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    f1 = (1.0 + f_beta_sq) * precision * recall / (f_beta_sq * precision + recall)

    precision[tp == 0] = 1.0
    recall[tp == 0] = 0.0
    f1[tp == 0] = 0.0

    return accuracy, precision, recall, f1


def multi_calc_confusion_matrix(gold, prediction):
    """Compute the confusion matrices of multiple models.
    :param gold -- the gold standard annotations of each model.
    :param prediction -- the system predictions in each model.
    """
    yy = np.sum((gold * prediction.T).T, axis=0)
    ny = np.sum(((1-gold) * prediction.T).T, axis=0)
    yn = np.sum((gold * (1-prediction.T)).T, axis=0)
    nn = np.sum(((1-gold) * (1-prediction.T)).T, axis=0)
    return yy, ny, yn, nn
