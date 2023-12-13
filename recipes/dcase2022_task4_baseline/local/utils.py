import os
from pathlib import Path
from typing import Optional, Union, List

import pandas as pd
import scipy
import torch
import numpy as np

from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics
import json

import soundfile
import glob


def batched_decode_preds(
    strong_preds_batch,
    weak_preds_batch,
    filenames,
    encoder,
    thresholds=[0.5],
    median_filter: Union[List[int], int] = 7,
    decode_weak: Optional[int] = None,
    pad_indicies=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary
    Args:
        strong_preds_batch: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int or list, the number of frames for which to apply median window (smoothing). (int: fixed, list: class-wise)
        decode_weak: int or None. flag to choose which method to use for utilizing weak prediction. (0: no weak prediction used, 1: weak prediction masking, 2: weak SED)
        pad_indicies: torch.Tensor, batch of indices which have been used for padding. (one index element is in range (0, 1])

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    pred_dict_by_thres = {}
    for threshold in thresholds:
        pred_dict_by_thres[threshold] = pd.DataFrame()

    for batch_indx in range(strong_preds_batch.shape[0]):  # over batches
        for c_th in thresholds:
            strong_preds = strong_preds_batch[batch_indx]
            if pad_indicies is not None:
                true_len = int(strong_preds.shape[-1] * pad_indicies[batch_indx].item())
                strong_preds = strong_preds[:true_len]
            strong_preds = strong_preds.transpose(0, 1).detach().cpu().numpy()  # size = (frames, n_class)
            if decode_weak in [1, 2]:
                for class_indx in range(weak_preds_batch.size(1)):
                    if weak_preds_batch[batch_indx, class_indx] < c_th:
                        strong_preds[:, class_indx] = 0
                    elif decode_weak >= 2:  # decode_weak == 2
                        strong_preds[:, class_indx] = 1
            if decode_weak is None or decode_weak < 2:  # decode_weak == None or 0 or 1
                strong_preds = strong_preds > c_th

                if type(median_filter) == int:
                    strong_preds = scipy.ndimage.filters.median_filter(strong_preds, (median_filter, 1))
                elif type(median_filter) == list:  # apply class-wise median filter
                    frames_list = [
                        scipy.ndimage.filters.median_filter(strong_preds[:, indx][:, np.newaxis], (filt_len, 1))
                        for indx, filt_len in enumerate(median_filter)
                    ]
                    strong_preds = np.hstack(frames_list)
                else:
                    raise Exception("unknown type of median_filter")

            strong_preds = encoder.decode_strong(strong_preds)
            strong_preds = pd.DataFrame(strong_preds, columns=["event_label", "onset", "offset"])
            strong_preds["filename"] = Path(filenames[batch_indx]).stem + ".wav"
            pred_dict_by_thres[c_th] = pred_dict_by_thres[c_th].append(strong_preds, ignore_index=True)

    return pred_dict_by_thres


def convert_to_event_based(weak_dataframe):
    """Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
    ('filename', 'onset', 'offset', 'event_label').

    Args:
        weak_dataframe: pd.DataFrame, the dataframe to be converted.

    Returns:
        pd.DataFrame, the dataframe strongly labeled.
    """

    new = []
    for i, r in weak_dataframe.iterrows():

        events = r["event_labels"].split(",")
        for e in events:
            new.append({"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1})
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


def parse_jams(jams_list, encoder, out_json):

    if len(jams_list) == 0:
        raise IndexError("jams list is empty ! Wrong path ?")

    backgrounds = []
    sources = []
    for jamfile in jams_list:

        with open(jamfile, "r") as f:
            jdata = json.load(f)

        # check if we have annotations for each source in scaper
        assert len(jdata["annotations"][0]["data"]) == len(
            jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"]
        )

        for indx, sound in enumerate(jdata["annotations"][0]["data"]):
            source_name = Path(jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"][indx]).stem
            source_file = os.path.join(
                Path(jamfile).parent,
                Path(jamfile).stem + "_events",
                source_name + ".wav",
            )

            if sound["value"]["role"] == "background":
                backgrounds.append(source_file)
            else:  # it is an event
                if sound["value"]["label"] not in encoder.labels:  # correct different labels
                    if sound["value"]["label"].startswith("Frying"):
                        sound["value"]["label"] = "Frying"
                    elif sound["value"]["label"].startswith("Vacuum_cleaner"):
                        sound["value"]["label"] = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                sources.append(
                    {
                        "filename": source_file,
                        "onset": sound["value"]["event_time"],
                        "offset": sound["value"]["event_time"] + sound["value"]["event_duration"],
                        "event_label": sound["value"]["label"],
                    }
                )

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"backgrounds": backgrounds, "sources": sources}, f, indent=4)


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """
        Generate a dataframe with filename and duration of the file

    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        pd.DataFrame: the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df
