import numpy as np
from sklearn import metrics


def get_best_f1(label, score):
    precision, recall, ths = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    best_th = ths[np.argmax(f1)]
    return best_f1, best_p, best_r, best_th


def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r, _ = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]

    Parameters
    ----------
        score: np.ndarray
            anomaly score, higher score indicates higher likelihoods to be anomaly
        label: np.ndarray
            ground-truth label

    Return
    ----------
        score: np.ndarray
            adjusted score

    """

    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score

