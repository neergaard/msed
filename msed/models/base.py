import json
import os

# import shutil
# import sys
import tarfile
import tempfile
from collections import OrderedDict

# from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import tqdm
import xmltodict

# from torch.autograd import Variable
# from torch.nn.modules.module import _addindent

from msed.functions import Detection
from msed.functions import binary_to_array
from msed.functions.metrics import delta_duration_function


class BaseNet(nn.Module):
    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()
        self.__dict__.update(kwargs)
        self.padding = self.kernel_size // 2
        self.detection_parameters["n_classes"] = self.n_classes
        self.detector = Detection(**self.detection_parameters)
        # self.localizations_default = nn.Parameter(torch.tensor(self.get_overlapping_default_events(), device=self.device), requires_grad=False)
        self.localizations_default = self.get_overlapping_default_events()

        # Possibly dropout
        if hasattr(self, "dropout") and self.dropout:  # self.dropout:
            self.dropout_layer = nn.Dropout2d(p=self.dropout)

    # def map2device(self):
    #     if torch.cuda.device_count() > 1 and

    def predict(self, x):
        # localizations, classifications, localizations_default = self.forward(x)
        localizations, classifications = self.forward(x)
        if not torch.is_tensor(self.localizations_default_expanded):
            localizations_default = torch.tensor(self.localizations_default_expanded).to(self.device)
        else:
            localizations_default = self.localizations_default_expanded

        return self.detector(localizations, classifications, localizations_default)

    def save(self, filename, checkpoint):
        torch.save(checkpoint, filename + ".tar")
        return filename

    @classmethod
    def load(cls, filename):
        with tarfile.open(filename, "r") as (tar):
            net_parameters = json.loads(tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict(torch.load(path + "/state.torch"))
        return (net, net_parameters)

    def predict_dataset(self, inference_dataset, threshold, overlap_factor=0.5, batch_size=128):
        """
        Predicts events in inference_dataset.
        """
        self.eval()
        if not isinstance(threshold, dict):
            self.detector.classification_threshold = threshold
        window_size = inference_dataset.window_size
        window = inference_dataset.window
        overlap = window * overlap_factor
        predictions = {k1: {} for k1 in inference_dataset.records}
        if not isinstance(threshold, dict):
            bar_inference = tqdm.tqdm((inference_dataset.records), desc=f"[ PREDICT ] Thr: {threshold:.3f}")
        else:
            bar_inference = tqdm.tqdm(inference_dataset.records, desc=f"[ PREDICT ]")
        with torch.no_grad():
            for idx, record in enumerate(bar_inference):
                bar_inference.set_postfix({"record": record})
                # if idx < 82:
                #     continue
                # if record not in ['mros-visit1-aa1401.h5', 'mros-visit1-aa3386.h5']:
                #     continue

                # Read hypnogram
                def read_hypnogram(file_id):
                    visit = file_id.split("-")[1]
                    record_id = file_id.split(".")[0]
                    xml_file = (
                        os.path.join("data", "raw", "polysomnography", "annotations-events-nsrr", "visit1", record_id)
                        + "-nsrr.xml"
                    )
                    with open(xml_file, "r") as f:
                        xml = f.read()
                    epoch_length = int(xmltodict.parse(xml)["PSGAnnotation"]["EpochLength"])
                    annotations = xmltodict.parse(xml)["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"]
                    # hypnogram_raw = [
                    #     el for el in annotations if el['EventType'] == 'Stages|Stages']
                    hypnogram_raw = [
                        int(el["EventConcept"][-1:])
                        for el in annotations
                        for _ in range(int(float(el["Duration"]) // epoch_length))
                        if el["EventType"] == "Stages|Stages"
                    ]

                    # Stage 4 gets converted to N3
                    hypnogram = [3 if h == 4 else h for h in hypnogram_raw]

                    return hypnogram

                # hypnogram = [h for h in read_hypnogram(record) for _ in range(30 * self.fs)]
                try:
                    hypnogram = read_hypnogram(record)
                    sleep_onset = next((i for i, h in enumerate(hypnogram) if h), None)
                    sleep_offset = len(hypnogram) - next((i for i, h in enumerate(reversed(hypnogram)) if h), None)
                except FileNotFoundError:
                    hypnogram = np.zeros((0,))
                predictions[record]["hypnogram"] = np.array(hypnogram)
                result = np.zeros(
                    (inference_dataset.n_classes, inference_dataset.metadata[record]["size"]), dtype=np.int8
                )
                # record_dataloader = inference_dataset.get_record_dataset(record, batch_size=int(batch_size), stride=overlap)
                record_dataloader = inference_dataset.get_record_dataset(record, batch_size=1, stride=overlap)
                for signals, times in record_dataloader:
                    # for signals, times in inference_dataset.get_record_batch(record,
                    #                                                          batch_size=int(batch_size),
                    #                                                          stride=overlap):
                    x = signals.to(self.device)
                    if isinstance(threshold, dict):
                        batch_predictions = []
                        for (event, event_thr), event_num in zip(threshold.items(), [0, 1, 2]):
                            self.detector.classification_threshold = event_thr
                            try:
                                event_predictions = self.predict(x)
                            except RuntimeError:
                                print("Bug")
                                event_predictions = self.predict(x)
                            for events, time in zip(event_predictions, times):
                                for ev in events:
                                    if ev[2] == event_num:
                                        start = int(round(ev[0] * window_size + time[0]))
                                        stop = int(round(ev[1] * window_size + time[0]))
                                        result[ev[2], start:stop] = 1
                    else:
                        try:
                            batch_predictions = self.predict(x)
                        except RuntimeError:
                            print("Bug")
                        for events, time in zip(batch_predictions, times):
                            for event in events:
                                start = int(round(event[0] * window_size + time[0]))
                                stop = int(round(event[1] * window_size + time[0]))
                                result[event[2], start:stop] = 1
                predicted_events = [binary_to_array(k) for k in result]
                assert len(predicted_events) == self.n_classes - 1
                for event_num in range(self.n_classes - 1):

                    # Screen events
                    if event_num == 0:
                        event = "arousal"
                        # screened_events = [ev for ev in predicted_events[event_num] if hypnogram[ev[0]] != 0 or hypnogram[ev[1]] != 0]
                        # Remove arousals that begin in wake where the next epoch is also wake
                        arousals = predicted_events[event_num]
                        screened_events = np.array(arousals)
                        # screened_events = [arousal for arousal in arousals
                        #    if (hypnogram[arousal[0] // (self.fs * 30)] != 0
                        #    or (hypnogram[arousal[0] // (self.fs * 30)] == 0 and hypnogram[arousal[0] // (self.fs * 30) + 1] != 0))
                        #    and (arousal[1] - arousal[0]) / self.fs >= 3.0
                        #    if arousal[0] // (self.fs * 30) >= sleep_onset
                        #    and arousal[0] // (self.fs * 30) <= sleep_offset
                        # ]

                    elif event_num == 1:
                        event = "lm"
                        # Remove limb movements shorter than 0.5 s and longer than 10 s not in wake
                        lms = predicted_events[event_num]
                        screened_events = np.array(lms)
                        # screened_events = [lm for lm in lms
                        #    if (set(hypnogram[lm[0] // (self.fs * 30) : lm[1] // (self.fs * 30) + 1]) != {0})
                        #    if ((lm[1] - lm[0]) / self.fs >= 0.5 and (lm[1] - lm[0]) / self.fs <= 10.0)
                        #    if lm[0] // (self.fs * 30) >= sleep_onset
                        #    and lm[0] // (self.fs * 30) <= sleep_offset]
                        # ]

                    elif event_num == 2:
                        event = "sdb"
                        # Note 3 concerning apneas: If an apnea/hypopnea occurs entirely during wake it is not counted in the AHI.
                        sdbs = predicted_events[event_num]
                        screened_events = np.array(sdbs)
                        # screened_events = [sdb for sdb in sdbs
                        #    if (set(hypnogram[sdb[0] // (self.fs * 30) : sdb[1] // (self.fs * 30) + 1]) != {0})
                        #    and (sdb[1] - sdb[0]) // self.fs >= 10.0
                        #    if sdb[0] // (self.fs * 30) >= sleep_onset
                        #    and sdb[0] // (self.fs * 30) <= sleep_offset
                        # ]

                    predictions[record][event] = screened_events

                # This is mainly for debugging
                # for event_num in range(len(predictions[record])):
                #     if len(predictions[record][event_num]) == 0:
                #         continue
                #     events = inference_dataset.get_record_events(record)[event_num]
                #     if len(events) == 0:
                #         continue
                #     m = delta_duration_function()(predictions[record][event_num],
                #                                 events,
                #                                 min_iou=0.1)

        return predictions

    @property
    def nelement(self):
        cpt = 0
        for p in self.parameters():
            cpt += p.nelement()

        return cpt

    def get_overlapping_default_events(self, window_size=None, default_event_sizes=None, factor_overlap=None):
        if not window_size:
            window_size = self.window_size
        if not default_event_sizes:
            default_event_sizes = [default_event_size * self.fs for default_event_size in self.default_event_sizes]
        if not factor_overlap:
            factor_overlap = self.factor_overlap
        window_size = window_size
        default_event_sizes = default_event_sizes
        factor_overlap = factor_overlap
        default_events = []
        for default_event_size in default_event_sizes:
            overlap = default_event_size / factor_overlap
            number_of_default_events = int(window_size / overlap)
            default_events.extend(
                [
                    (overlap * (0.5 + i) / window_size, default_event_size / window_size)
                    for i in range(number_of_default_events)
                ]
            )
        #         return default_events.to(self.device)
        # return default_events

        return np.array(default_events, dtype=np.float32)

    def print_info_architecture(self, fs=None):
        if fs is None:
            fs = self.fs
        size = self.window_size
        receptive_field = 0
        print("\nInput feature map size: {}".format(size))
        print("Input receptive field: {}".format(receptive_field))
        print("Input size in seconds: {} s".format(size / fs))
        print("Input receptive field in seconds: {} s \n".format(receptive_field / fs))
        kernal_size = self.kernel_size
        size //= 2
        receptive_field = kernal_size + 1
        print("After layer 1:")
        print("\tFeature map size: {}".format(size))
        print("\tReceptive field: {}".format(receptive_field))
        print("\tReceptive field in seconds: {} s".format(receptive_field / fs))
        for layer in range(2, self.k_max + 1):
            size //= 2
            receptive_field += kernal_size // 2 * 2 * 2 ** (layer - 1)
            receptive_field += 2 ** (layer - 1)
            print("After layer {}:".format(layer))
            print("\tFeature map size: {}".format(size))
            print("\tReceptive field: {}".format(receptive_field))
            print("\tReceptive field in seconds: {} s".format(receptive_field / fs))

        print("\n")

    def summary(self, input_size, batch_size=-1):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[(-1)].split("'")[0]
                if class_name in ["Sequential", "BottleneckBlock", "AdditiveAttention", "Stream"]:
                    return
                module_idx = len(summary)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    if class_name in ("GRU", "LSTM", "RNN"):
                        summary[m_key]["output_shape"] = [batch_size] + list(output[0].size())[1:]
                    else:
                        summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                summary[m_key]["trainable"] = any([p.requires_grad for p in module.parameters()])
                params = np.sum([np.prod(list(p.size())) for p in module.parameters() if p.requires_grad])
                summary[m_key]["nb_params"] = int(params)

            if not isinstance(module, nn.Sequential):
                if not isinstance(module, nn.ModuleList):
                    pass
            if not module == self:
                hooks.append(module.register_forward_hook(hook))

        # device = self.device.lower()
        # assert device in ('cuda', 'cpu'), "Input device is not valid, please specify 'cuda' or 'cpu'"
        # if device == 'cuda':
        # if torch.cuda.is_available():
        #     dtype = torch.cuda.FloatTensor
        # else:
        dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [(torch.rand)(*(2,), *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        self.apply(register_hook)
        self(*x)
        for h in hooks:
            h.remove()

        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"])
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"]:
                    trainable_params += summary[layer]["nb_params"]
                print(line_new)

        total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / float(1024**2))
        total_output_size = abs(2.0 * total_output * 4.0 / float(1024**2))
        total_params_size = abs(total_params * 4.0 / float(1024**2))
        total_size = total_params_size + total_output_size + total_input_size
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print(f"Forward/backward pass size (GB): {abs(2.0 * total_output * 4.0 / float(1024 ** 3)):0.2f}")
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print(
            f"Estimated Total Size (GB): {(total_params + 2.0 * total_output + np.prod(input_size) * batch_size) * 4.0 / float(1024 ** 3):0.2f}"
        )
        print("----------------------------------------------------------------")
