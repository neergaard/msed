import json
import re
from pathlib import Path
from typing import List, Optional, Union

import einops
import mne
import numpy as np
import pandas as pd
import torch

from msed.functions import binary_to_array
from msed.models import network as network_models
from msed.preprocessing import process_file, get_channel_mapper, FS
from msed.utils.argument_parser import check_and_return_args
from msed.utils.config import load_config, Config
from msed.utils.logger import get_logger
from msed.utils.misc import check_and_return_device


logger = get_logger()


def get_datapaths(
    data_path: Path,
    pattern: Optional[str] = None,
) -> Union[List[Path], pd.DataFrame]:

    logger.info(f'Getting relevant EDF files from "{data_path}"...')
    # Get data paths
    if data_path.is_dir():
        data_list = sorted(list(data_path.rglob("*.[EeRr][DdEe][FfCc]")))
    else:
        data_list = []

    # Maybe trim according to specificied pattern
    if pattern is not None:
        data_list = [p for p in data_list if bool(re.search(pattern, p.stem) and p.suffix.lower() in [".edf", ".rec"])]

    # This is for bookkeeping
    n_files = len(data_list)
    df = pd.DataFrame(
        {
            "ID": [p.stem for p in data_list],
            "Path": data_list,
            "Success": [False] * n_files,
            "No. events": [np.nan] * n_files,
        }
    )
    logger.info(f"Found {n_files} EDF files.")

    return data_list, df


def initialize_model(config: Config) -> torch.nn.Module:

    # specificy additional parameters
    fs = config.dataset["args"]["signals"]["fs"]
    window_duration = config.dataset["args"]["window"]
    window_size = int(fs * window_duration)
    n_classes = len(config.dataset["args"]["events"]) + 1
    additional_params = {
        "n_channels": config.dataset["args"]["n_channels"],
        "n_classes": n_classes,
        "window_size": window_size,
    }
    network_type = config.network["type"]
    network_args = config.network["args"]
    network_args.update(additional_params)
    # HACK: remove the device spec
    network_args.pop("device", None)

    # Create the model
    model = getattr(network_models, network_type)(**network_args)

    # update the detection thresholds
    model.detector.class_key = {1: "arousal", 2: "lm", 3: "sdb"}
    model.detector.classification_threshold = config.threshold
    return model


def get_model_from_config(ckpt_path: Path, config, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path / "weights.pth", map_location="cpu", weights_only=True)
    model = initialize_model(config)
    model.load_state_dict(ckpt)
    model.to(device)
    model.device = device
    return model


def predict_events(
    model_path: Path, target_dir: Path, data_path: Path, pattern: Optional[str] = None, device: str = "cpu"
):
    logger.info(f'Loading model configuration from "{model_path}"...')
    config = load_config(model_path)

    # Determine data paths
    data_paths, df = get_datapaths(data_path=data_path, pattern=pattern)

    # Create the channel map object
    logger.info("Creating channel map (if it does not exist)...")
    channel_map = get_channel_mapper(data_paths)

    # Get model and device
    logger.info(f'Loading model from "{model_path}"...')
    device = check_and_return_device(device)
    model = get_model_from_config(model_path, config, device)
    logger.info(f'Model initialized on device "{device}".')

    # Run over data files
    class_names = list(config.threshold.keys())
    for ev in class_names:
        df[f"{ev}"] = [np.nan] * len(df)
    logger.info(f"Class names: {class_names}")
    predictions = {k: {ev: [] for ev in class_names} for k in df["ID"]}
    model.eval()
    with torch.no_grad():
        logger.info(f"Running inference over {len(data_paths)} files...")

        for idx, (_, row) in enumerate(df.iterrows()):
            subject_id = row["ID"]
            data_path = row["Path"]

            # load and process EDF
            try:
                logger.info(f"[ {subject_id} ] Running preprocessing ...")
                data = process_file(data_path, channel_map)
            except Exception as e:
                logger.error(f"Error processing {data_path}: {e}")
                continue

            # run inference
            logger.info(f"[ {subject_id} ] Running inference ...")
            window_size = model.window_size
            stride = int(0.5 * window_size)
            x = einops.rearrange(
                torch.tensor(data, dtype=torch.float32, device=device).unfold(-1, window_size, stride),
                "C N T -> N C T",
            )
            preds = model.predict(x)

            # convert to predictions masks
            n_events = []
            prediction_mask = np.zeros((len(config.threshold), data.shape[-1]), dtype=np.uint8)
            for window_idx, window_events in enumerate(preds):
                for ev in window_events:
                    start = int(ev[0] * window_size) + window_idx * stride
                    stop = int(ev[1] * window_size) + window_idx * stride
                    prediction_mask[ev[-1] - 1, start:stop] = 1
            for ev, p in zip(class_names, prediction_mask):
                predictions[subject_id][ev] = binary_to_array(p)
                n_event = len(predictions[subject_id][ev])
                df.loc[idx, f"{ev}"] = int(n_event)
                n_events.append(n_event)
                logger.info(f'[ {subject_id} ] Found {n_event} "{ev}" events.')
            df.loc[idx, "Events"] = int(sum(n_events))
            logger.info(f"[ {subject_id} ] Found {sum(n_events)} events in total.")

            # Save predictions for single subject in csv file
            df.loc[idx, "Success"] = True
            subject_pred_path = target_dir / f"{subject_id}.csv"
            logger.info(f'[ {subject_id} ] Saving predictions at "{subject_pred_path}"...')
            event_start_idx, event_stop_idx, duration_idx, event_start, event_stop, duration, classes = zip(
                *[
                    (ev[0], ev[1], ev[1] - ev[0], ev[0] / FS, ev[1] / FS, ev[1] / FS - ev[0] / FS, class_name)
                    for class_name, events in predictions[subject_id].items()
                    for ev in events
                ]
            )
            pd.DataFrame({"Start_sample": event_start_idx, "Stop_sample": event_stop_idx, "Duration_sample": duration_idx, "Start_second": event_start, "Stop_second": event_stop, "Duration_second": duration, "Class": classes}).to_csv(
                subject_pred_path, index=False
            )

    # Save status to file
    sumamary_path = target_dir / "status.csv"
    df.to_csv(sumamary_path, index=False)
    logger.info(f'Done! Check "{sumamary_path}" for details.')


def main_cli():
    args = check_and_return_args()
    predict_events(args.model_path, args.target_dir, args.data_path, args.match_pattern, args.device)


if __name__ == "__main__":
    main_cli()
