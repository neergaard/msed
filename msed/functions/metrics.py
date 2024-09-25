import numpy as np
import torch

from msed.functions import jaccard_overlap


def precision_function():
    """returns a function to calculate precision"""

    def calculate_precision(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the precision.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        # iou = jaccard_overlap(torch.Tensor(prediction), torch.Tensor(reference))
        iou = jaccard_overlap(prediction, reference)
        max_iou = iou.max(axis=1)
        # max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum()
        # true_positive = (max_iou >= min_iou).sum().item()
        false_positive = len(prediction) - true_positive
        precision = true_positive / (true_positive + false_positive)
        return precision

    return calculate_precision


def recall_function():
    """returns a function to calculate recall"""

    def calculate_recall(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the recall.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        iou = jaccard_overlap(prediction, reference)
        # iou = jaccard_overlap(torch.Tensor(prediction), torch.Tensor(reference))
        max_iou = iou.max(axis=1)
        # max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum()
        # true_positive = (max_iou >= min_iou).sum().item()
        false_negative = len(reference) - true_positive
        recall = true_positive / (true_positive + false_negative)
        return recall

    return calculate_recall


def f1_function():
    """returns a function to calculate f1 score"""

    def calculate_f1_score(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the f1 score.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        iou = jaccard_overlap(prediction, reference)
        # iou = jaccard_overlap(torch.Tensor(prediction), torch.Tensor(reference))
        max_iou = iou.max(axis=1)
        # max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum()
        # true_positive = (max_iou >= min_iou).sum().item()
        false_positive = len(prediction) - true_positive
        false_negative = len(reference) - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision == 0 or recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    return calculate_f1_score


def delta_duration_function():

    def calculate_delta_duration(prediction, reference, min_iou=0.3, fs=128):

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        iou = jaccard_overlap(prediction, reference)
        max_iou = iou.max(axis=1)

        # In case there's zero overlap between any predictions and targets
        if all(max_iou == 0.0):
            return np.nan

        prediction_idx = np.where(max_iou >= min_iou)[0]
        reference_idx = np.array(
            [
                np.where(row_max_iou == row_iou)[0]
                for row_max_iou, row_iou in zip(max_iou, iou)
                if row_max_iou >= min_iou
            ]
        ).squeeze()

        # If there's some overlap but not any above threshold
        if prediction_idx.size == 0 or reference_idx.size == 0:
            return np.nan

        # If a prediction is matched to multiple targets, we compute the statistics wrt. each target
        if reference_idx.dtype != np.int64:
            while True:
                for idx, row in enumerate(reference_idx):
                    if isinstance(row, np.ndarray) and len(row) > 1:
                        for i in range(len(reference_idx[idx]) - 1):
                            prediction_idx = np.insert(prediction_idx, idx, prediction_idx[idx])
                        for k, v in enumerate(reference_idx[idx]):
                            reference_idx = np.insert(reference_idx, idx + k + 1, v)
                        reference_idx = np.delete(reference_idx, idx)
                        break
                try:
                    if idx == len(reference_idx) - 1:
                        break
                except:
                    print("bug")
            reference_idx = np.array([np.squeeze(a) for a in reference_idx])
        duration_prediction = np.diff(prediction[prediction_idx])
        duration_reference = np.diff(reference[reference_idx])
        delta_duration = (duration_prediction - duration_reference) / fs

        return delta_duration

    return calculate_delta_duration


def delta_start_function():

    def calculate_delta_start(prediction, reference, min_iou=0.3, fs=128):

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        iou = jaccard_overlap(prediction, reference)
        max_iou = iou.max(axis=1)

        # In case there's zero overlap between any predictions and targets
        if all(max_iou == 0.0):
            return np.nan

        prediction_idx = np.where(max_iou >= min_iou)[0]
        reference_idx = np.array(
            [
                np.where(row_max_iou == row_iou)[0]
                for row_max_iou, row_iou in zip(max_iou, iou)
                if row_max_iou >= min_iou
            ]
        ).squeeze()

        # If there's some overlap but not any above threshold
        if prediction_idx.size == 0 or reference_idx.size == 0:
            return np.nan

        if reference_idx.dtype != np.int64:
            while True:
                for idx, row in enumerate(reference_idx):
                    if isinstance(row, np.ndarray) and len(row) > 1:
                        for i in range(len(reference_idx[idx]) - 1):
                            prediction_idx = np.insert(prediction_idx, idx, prediction_idx[idx])
                        for k, v in enumerate(reference_idx[idx]):
                            reference_idx = np.insert(reference_idx, idx + k + 1, v)
                        reference_idx = np.delete(reference_idx, idx)
                        break
                if idx == len(reference_idx) - 1:
                    break
            reference_idx = np.array([np.squeeze(a) for a in reference_idx])
        start_prediction = prediction[prediction_idx, 0]
        start_reference = reference[reference_idx, 0]
        delta_start = (start_prediction - start_reference) / fs

        return delta_start

    return calculate_delta_start


def delta_stop_function():

    def calculate_delta_stop(prediction, reference, min_iou=0.3, fs=128):

        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=np.float32)
        if isinstance(reference, list):
            reference = np.array(reference, dtype=np.float32)

        iou = jaccard_overlap(prediction, reference)
        max_iou = iou.max(axis=1)

        # In case there's zero overlap between any predictions and targets
        if all(max_iou == 0.0):
            return np.nan

        prediction_idx = np.where(max_iou >= min_iou)[0]
        reference_idx = np.array(
            [
                np.where(row_max_iou == row_iou)[0]
                for row_max_iou, row_iou in zip(max_iou, iou)
                if row_max_iou >= min_iou
            ]
        ).squeeze()

        # If there's some overlap but not any above threshold
        if prediction_idx.size == 0 or reference_idx.size == 0:
            return np.nan

        if reference_idx.dtype != np.int64:
            while True:
                for idx, row in enumerate(reference_idx):
                    if isinstance(row, np.ndarray) and len(row) > 1:
                        for i in range(len(reference_idx[idx]) - 1):
                            prediction_idx = np.insert(prediction_idx, idx, prediction_idx[idx])
                        for k, v in enumerate(reference_idx[idx]):
                            reference_idx = np.insert(reference_idx, idx + k + 1, v)
                        reference_idx = np.delete(reference_idx, idx)
                        break
                if idx == len(reference_idx) - 1:
                    break
            reference_idx = np.array([np.squeeze(a) for a in reference_idx])

        stop_prediction = prediction[prediction_idx, 1]
        stop_reference = reference[reference_idx, 1]
        delta_stop = (stop_prediction - stop_reference) / fs

        return delta_stop

    return calculate_delta_stop
