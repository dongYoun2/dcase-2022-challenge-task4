import argparse
import re
import yaml
import warnings
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from typing import List
import os
from glob import glob

from desed_task.utils.encoder import ManyHotEncoder

from local.classes_dict import classes_labels
from local.utils import batched_decode_preds, log_sedeval_metrics

from desed_task.evaluation.evaluation_measures import (
    compute_psds_from_operating_points,
    compute_per_intersection_macro_f1,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_scores(score_path: str):
    scores = np.load(score_path, allow_pickle=True)

    file_dict = {}
    for ndarr in scores:
        file_dict[ndarr["filename"]] = ndarr

    to_tensor = lambda preds: torch.Tensor(np.array(preds)).unsqueeze(0)
    for k in sorted(file_dict):
        strong_preds_student = to_tensor(file_dict[k]["student"])
        weak_preds_student = to_tensor(file_dict[k]["student_weak"])
        strong_preds_teacher = to_tensor(file_dict[k]["teacher"])
        weak_preds_teacher = to_tensor(file_dict[k]["teacher_weak"])

        yield [
            file_dict[k]["filename"]
        ], strong_preds_student, weak_preds_student, strong_preds_teacher, weak_preds_teacher


def decode(score_dict_tuple):
    global threshold_list, median_filters, decode_weak_mode, encoder

    filenames = score_dict_tuple[0][0]
    weak_preds, strong_preds = [], []
    for _, sps, wps, spt, wpt in score_dict_tuple:
        weak_preds.extend([wps, wpt])
        strong_preds.extend([sps, spt])
    weak_preds_mean = torch.mean(torch.stack(weak_preds), dim=0)
    strong_preds_mean = torch.mean(torch.stack(strong_preds), dim=0)

    # for psds1, psds2
    decoded_strong_per_thres = batched_decode_preds(
        strong_preds_mean,
        weak_preds_mean,
        filenames,
        encoder,
        thresholds=threshold_list,
        median_filter=median_filters,
        decode_weak=decode_weak_mode,
    )

    # for collar-based f1, intersection-based f1
    f1_score_thres = 0.5
    decoded_strong = batched_decode_preds(
        strong_preds_mean,
        weak_preds_mean,
        filenames,
        encoder,
        thresholds=[f1_score_thres],
        median_filter=median_filters,
        decode_weak=decode_weak_mode,
    )
    return decoded_strong_per_thres, decoded_strong[f1_score_thres]


def main(config, score_paths: List[str], mode: int):
    global threshold_list, median_filters, decode_weak_mode, encoder

    postprocess_config = config["postprocess"]
    thres_num = postprocess_config["n_test_thresholds"]
    threshold_list = list(np.arange(1 / (thres_num * 2), 1, 1 / thres_num))
    decode_weak_mode = postprocess_config["decode_weak"]
    median_filters = postprocess_config["median_window"]

    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    test_psds_buffer = defaultdict(pd.DataFrame)
    test_f1_buffer = pd.DataFrame()
    with mp.Pool(processes=mp.cpu_count() // 2) as pool, tqdm(
        total=len(list(load_scores(score_paths[0]))), desc="decoding"
    ) as pbar:
        for decoded_strong_per_thres, decoded_strong in tqdm(
            pool.imap(decode, zip(*[load_scores(p) for p in score_paths]))
        ):
            for th in threshold_list:
                test_psds_buffer[th] = test_psds_buffer[th].append(decoded_strong_per_thres[th], ignore_index=True)
            test_f1_buffer = test_f1_buffer.append(decoded_strong)
            pbar.update()

    if mode == 0:
        save_dir = config["submission"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        test_f1_buffer.to_csv(os.path.join(save_dir, "predictions_05.tsv"), sep="\t", index=False)

        for k in test_psds_buffer.keys():
            test_psds_buffer[k].to_csv(
                os.path.join(save_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )
        print(f"\nPredictions for student saved in: {save_dir}")

    elif mode in [1, 2]:
        if mode == 1:
            test_config = config["test"]
            save_dir = test_config.get("save_dir", None)
            tsv_file = test_config["test_tsv"]
            dur_file = test_config["test_dur"]
        else:
            public_eval_config = config["public_eval"]
            save_dir = public_eval_config.get("save_dir", None)
            tsv_file = public_eval_config["public_eval_tsv"]
            dur_file = public_eval_config["public_eval_dur"]
        psds1 = compute_psds_from_operating_points(
            test_psds_buffer,
            tsv_file,
            dur_file,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "psds1"),
        )

        psds2 = compute_psds_from_operating_points(
            test_psds_buffer,
            tsv_file,
            dur_file,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "psds2"),
        )

        event_f1_macro = log_sedeval_metrics(test_f1_buffer, tsv_file, save_dir=save_dir)[0]

        intersection_f1_macro = compute_per_intersection_macro_f1(
            {"0.5": test_f1_buffer},
            tsv_file,
            dur_file,
        )

        best_test_result = torch.tensor(max(psds1, psds2))

        results = {
            "hp_metric": best_test_result,
            "PSDS1": psds1,
            "PSDS2": psds2,
            "event_f1_macro": event_f1_macro,
            "intersection_f1_macro": intersection_f1_macro,
        }

        print(results)


def parse_score_paths(from_epoch: int, to_epoch: int, step: int, score_arr_pattern: str) -> List[str]:
    score_paths = glob(score_arr_pattern)
    parse_epoch = lambda path: int(path.split("epoch=")[1].split("-")[0])
    score_paths = [p for p in sorted(score_paths, key=parse_epoch) if from_epoch <= parse_epoch(p) <= to_epoch]
    score_paths = score_paths[::step]

    return score_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", default="./confs/postprocess.yaml")
    parser.add_argument("--mode", type=int, required=True)

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    if args.mode == 0:
        submission_config = config["submission"]
        score_arr_pattern = os.path.join(submission_config["system_dir"], "version_0/metrics_eval_eval21_*/scores.npy")
        score_paths = parse_score_paths(
            submission_config["from_epoch"],
            submission_config["to_epoch"],
            submission_config["step"],
            score_arr_pattern,
        )
    elif args.mode == 1:
        score_paths = [
            # "desed-lab/exps/e1000-adjust_batch/version_0/metrics_test_validation_epoch=164-step=54779/scores.npy",
            # "desed-lab/exps/add_specaug/version_0/metrics_test_validation_epoch=234-step=27729/scores.npy",
            # "desed-lab/exps/lrdecay/version_0/metrics_test_validation_epoch=214-step=25369/scores.npy",
            "desed-lab/exps/lrdecay-postprocess/version_0/metrics_test_validation_epoch=214-step=25369/scores.npy",
        ]
    elif args.mode == 2:
        public_eval_config = config["public_eval"]
        score_arr_pattern = os.path.join(public_eval_config["system_dir"], "version_0/metrics_test_public_*/scores.npy")
        score_paths = parse_score_paths(
            public_eval_config["from_epoch"],
            public_eval_config["to_epoch"],
            public_eval_config["step"],
            score_arr_pattern,
        )

    main(config, score_paths, args.mode)
