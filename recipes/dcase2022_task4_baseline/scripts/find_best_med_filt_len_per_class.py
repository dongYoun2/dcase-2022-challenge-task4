import argparse
import pytorch_lightning as pl
import yaml
import warnings
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from desed_task.utils.encoder import ManyHotEncoder
from typing import List

from local.sed_trainer import SEDTask4
from local.classes_dict import classes_labels
from local.utils import batched_decode_preds
from local import models

from desed_task.evaluation.evaluation_measures import (
    compute_psds_from_operating_points,
    compute_per_intersection_macro_f1,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_scores(score_path: str):
    scores = np.load(score_path, allow_pickle=True)
    batch_size = 24
    batch_num = int(np.ceil(len(scores) / batch_size))

    to_tensor = lambda preds: torch.Tensor(np.array(preds))
    for i in range(batch_num):
        batch_scores = scores[i * batch_size : (i + 1) * batch_size]
        filenames = []
        strong_preds_student, weak_preds_student = [], []
        strong_preds_teacher, weak_preds_teacher = [], []
        for score in batch_scores:
            strong_preds_student.append(score["student"])
            weak_preds_student.append(score["student_weak"])
            strong_preds_teacher.append(score["teacher"])
            weak_preds_teacher.append(score["teacher_weak"])
            filenames.append(score["filename"])

        strong_preds_student = to_tensor(strong_preds_student)
        weak_preds_student = to_tensor(weak_preds_student)
        strong_preds_teacher = to_tensor(strong_preds_teacher)
        weak_preds_teacher = to_tensor(weak_preds_teacher)

        yield filenames, strong_preds_student, weak_preds_student, strong_preds_teacher, weak_preds_teacher


def find_best_filter_len(
    config, threshold_num: int, score_path: str, class_num: int, min: List[int], max: List[int], step: List[int]
):
    best_med_filters = [7] * class_num
    threshold_list = list(np.arange(1 / (threshold_num * 2), 1, 1 / threshold_num))
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    for class_idx in tqdm(range(class_num), desc="processing by class_indx"):
        result_list = []
        for filt_value in range(min[class_idx], max[class_idx] + 1, step[class_idx]):
            filters = [f for f in best_med_filters]
            filters[class_idx] = filt_value

            test_psds_buffer_student = defaultdict(pd.DataFrame)
            for filenames, sps, wps, _, _ in load_scores(score_path):
                decoded_student_strong = batched_decode_preds(
                    sps, wps, filenames, encoder, thresholds=threshold_list, median_filter=filters, decode_weak=1
                )

                for th in threshold_list:
                    test_psds_buffer_student[th] = test_psds_buffer_student[th].append(
                        decoded_student_strong[th], ignore_index=True
                    )

            student_PSDS1 = compute_psds_from_operating_points(
                test_psds_buffer_student,
                config["data"]["test_tsv"],
                config["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
            )

            result_list.append((filt_value, student_PSDS1))

        result_list = sorted(result_list, key=lambda v: v[1])
        best_med_filt_len, best_psds1 = result_list[-1][0], result_list[-1][1]
        best_med_filters[class_idx] = best_med_filt_len
        print(f"candidates for class {class_idx}: ", result_list)
        print(
            f"best median filter length and student/PSDS1 score for for class {class_idx}: {best_med_filt_len}, {best_psds1}"
        )

    return best_med_filters


if __name__ == "__main__":
    CLASS_NUM = 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", default=None)
    parser.add_argument(
        "--score_path",
        type=str,
        default="desed-lab/exps/weak_pred_masking/version_1/metrics_test_validation_epoch=139-step=16519/scores.npy",
    )
    parser.add_argument("--min", nargs="+", type=int, default=[2 for _ in range(CLASS_NUM)])
    parser.add_argument("--max", nargs="+", type=int, default=[50 for _ in range(CLASS_NUM)])
    parser.add_argument("--step", nargs="+", type=int, default=[2 for _ in range(CLASS_NUM)])
    parser.add_argument("--n_test_thresholds", type=int, default=50)

    args = parser.parse_args()

    if len(args.min) != CLASS_NUM or len(args.max) != CLASS_NUM or len(args.step) != CLASS_NUM:
        raise Exception(f"min, max, or step argument length is not {CLASS_NUM}")

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    best_filters = find_best_filter_len(
        config, args.n_test_thresholds, args.score_path, CLASS_NUM, args.min, args.max, args.step
    )
    print(best_filters)
