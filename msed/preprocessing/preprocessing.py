import json
from pathlib import Path

import mne
import numpy as np

from msed.utils.channel_mapper import channel_mapper
from msed.utils.logger import get_logger

logger = get_logger()


eeg_eog_filter = dict(
    l_freq=0.3,
    h_freq=35,
    method="iir",
    iir_params={"output": "sos", "order": 2, "ftype": "butter", "btype": "bandpass"},
)
emg_filter = dict(
    l_freq=10,
    h_freq=None,
    method="iir",
    iir_params={"output": "sos", "order": 4, "ftype": "butter", "btype": "highpass"},
)
nasal_filter = dict(
    l_freq=0.03,
    h_freq=None,
    method="iir",
    iir_params={"output": "sos", "order": 4, "ftype": "butter", "btype": "highpass"},
)
belt_filter = dict(
    l_freq=0.1, h_freq=15, method="iir", iir_params={"output": "sos", "order": 2, "ftype": "butter", "btype": "band"}
)
channel_filters = {
    "C3": eeg_eog_filter,
    "C4": eeg_eog_filter,
    "EOGL": eeg_eog_filter,
    "EOGR": eeg_eog_filter,
    "Chin": emg_filter,
    "LegL": emg_filter,
    "LegR": emg_filter,
    "NasalP": nasal_filter,
    "Thor": belt_filter,
    "Abdo": belt_filter,
}


CHANNELS_TO_LOAD = [
    "A1",
    "A2",
    "C3",
    "C4",
    "EOGL",
    "EOGR",
    "EOGRef",
    "Chin",
    "ChinRef",
    "LegL",
    "LegR",
    "NasalP",
    "Thor",
    "Abdo",
]
CHANNEL_ORDER = ["C3", "C4", "EOGL", "EOGR", "Chin", "LegL", "LegR", "NasalP", "Thor", "Abdo"]
FS = 128


def process_file(data_path: Path, channel_map: dict):
    subject_id = data_path.stem
    header = mne.io.read_raw_edf(data_path, verbose=False).info

    # extract specific channels
    data = {k: None for k in CHANNELS_TO_LOAD}
    for ch_category in data.keys():
        for ch in channel_map[ch_category]:
            if ch in header.ch_names:
                data[ch_category] = mne.io.read_raw_edf(data_path, include=ch, preload=True, verbose=False)
                break

    # Check if all channels are present
    missing_channels = [k for k, v in data.items() if v is None]
    if len(set(CHANNEL_ORDER) - set(missing_channels)) != len(CHANNEL_ORDER):
        raise ValueError(f"Missing channels: {missing_channels}")

    # fmt: off
    # Potentially reference channels
    if data["A1"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "C4" to "A1"...')
        (data["C4"].add_channels([data["A1"]])
                   .set_eeg_reference(ref_channels=data["A1"].ch_names, verbose=False)
                   .drop_channels(data["A1"].ch_names))
    elif data["A2"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "C4" to "A2"...')
        (data["C4"].add_channels([data["A2"]])
                   .set_eeg_reference(ref_channels=data["A2"].ch_names, verbose=False)
                   .drop_channels(data["A2"].ch_names))
    if data["A2"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "C3" to "A2"...')
        (data["C3"].add_channels([data["A2"]])
                   .set_eeg_reference(ref_channels=data["A2"].ch_names, verbose=False)
                   .drop_channels(data["A2"].ch_names))
    elif data["A1"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "C3" to "A1"...')
        (data["C3"].add_channels([data["A1"]])
                   .set_eeg_reference(ref_channels=data["A1"].ch_names, verbose=False)
                   .drop_channels(data["A1"].ch_names))
    if data["EOGRef"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "EOGL" to "EOGRef"...')
        (data["EOGL"].add_channels([data["EOGRef"]])
                     .set_eeg_reference(ref_channels=data["EOGRef"].ch_names, verbose=False)
                     .drop_channels(data["EOGRef"].ch_names))
        logger.info(f'[ {subject_id} ] Referencing channel "EOGR" to "EOGRef"...')
        (data["EOGR"].add_channels([data["EOGRef"]])
                     .set_eeg_reference(ref_channels=data["EOGRef"].ch_names, verbose=False)
                     .drop_channels(data["EOGRef"].ch_names))
    if data["ChinRef"] is not None:
        logger.info(f'[ {subject_id} ] Referencing channel "Chin" to "ChinRef"...')
        (data["Chin"].add_channels([data["ChinRef"]])
                     .set_eeg_reference(ref_channels=data["ChinRef"].ch_names, verbose=False)
                     .drop_channels(data["ChinRef"].ch_names))
    [data.pop(ch, None) for ch in ["A1", "A2", "EOGRef", "ChinRef"]]
    # fmt: on

    # resample all signals to common sampling rate
    for chn in data.keys():
        logger.info(f'[ {subject_id} ] Resampling channel "{chn}"...')
        data[chn].resample(FS, method="polyphase", verbose=False)

    # filter all signals depending on the channel
    for chn in data.keys():
        logger.info(f'[ {subject_id} ] Filtering channel "{chn}"...')
        data[chn].filter(**channel_filters[chn], verbose=False)

    # Collect everything into a single tensor and standardize per channel
    data = np.stack([data[chn].get_data()[0] for chn in CHANNEL_ORDER])
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

    return data


def get_channel_mapper(data_paths):
    channel_map_path = data_paths[0].parent / "channel_map.json"
    if channel_map_path.exists():
        with open(channel_map_path, "r") as f:
            channel_map = json.load(f)
    else:
        channel_map = channel_mapper(data_paths, CHANNELS_TO_LOAD, channel_map_path)
    logger.info(f'Created channel map at "{channel_map_path}"')
    return channel_map
