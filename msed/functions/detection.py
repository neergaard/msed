import torch
import torch.nn as nn
import torch.nn.functional as F

from msed.functions import decode
from msed.functions.non_maximum_suppression import non_maximum_suppression


class Detection(nn.Module):
    """
    Inspired from https://github.com/amdegroot/ssd.pytorch
    This is a modified version of the Detection module from the DOSED repository.
    https://github.com/Dreem-Organization/dosed/blob/master/dosed/functions/detection.py
    """

    def __init__(
        self, n_classes, overlap_non_maximum_suppression, classification_threshold, class_key=None, softmax=True
    ):
        super().__init__()
        self.n_classes = n_classes
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.classification_threshold = classification_threshold
        self.class_key = class_key
        self.softmax = softmax
        assert self.class_key if isinstance(self.classification_threshold, dict) else True

    def forward(self, localizations, classifications, localizations_default):
        batch = localizations.size(0)
        if self.softmax:
            scores = F.softmax(classifications, dim=-1)  # nn.Softmax(dim=-1)(classifications)
        else:
            scores = torch.zeros_like(classifications)
            for class_index in range(1, self.n_classes):
                scores[:, :, class_index] = F.softmax(classifications[:, :, [0, class_index]], dim=-1)[:, :, 1]
            # scores = torch.sigmoid(classifications)
        results = []
        for i in range(batch):
            result = []
            localization_decoded = decode(localizations[i], localizations_default)
            for class_index in range(1, self.n_classes):
                if isinstance(self.classification_threshold, dict):
                    thr = self.classification_threshold[self.class_key[class_index]]
                else:
                    thr = self.classification_threshold
                scores_batch_class = scores[i, :, class_index]
                scores_batch_class_selected = scores_batch_class[(scores_batch_class > thr)]
                if len(scores_batch_class_selected) == 0:
                    continue
                localizations_decoded_selected = localization_decoded[
                    (scores_batch_class > thr).unsqueeze(1).expand_as(localization_decoded)
                ].view(-1, 2)
                events = non_maximum_suppression(
                    localizations_decoded_selected,
                    scores_batch_class_selected,
                    overlap=self.overlap_non_maximum_suppression,
                )
                result.extend([(event[0].item(), event[1].item(), class_index - 1) for event in events])

            results.append(result)

        return results
