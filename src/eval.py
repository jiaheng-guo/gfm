def precision_score(tp, fp):
    """Compute precision score.

    Parameters
    ----------
        tp: int
            True positives
        fp: int
            False positives

    Returns
    -------
        precision: float
            Precision score
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)