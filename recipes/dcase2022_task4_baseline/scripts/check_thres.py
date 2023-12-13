import argparse
import pytorch_lightning as pl
import yaml
import warnings
import pandas as pd
import numpy as np

from desed_task.utils.encoder import ManyHotEncoder
from desed_task.dataio.datasets import StronglyAnnotatedSet

from local.sed_trainer import SEDTask4
from local.classes_dict import classes_labels
from local.modes import TEST
from local.utils import batched_decode_preds
from local import models

from desed_task.evaluation.evaluation_measures import (
    compute_psds_from_operating_points,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class SEDTask4Custom(SEDTask4):
    def set_threshold(self, thres):
        self.threshold = thres

    def on_test_start(self):
        self.test_psds_buffer_student = {self.threshold: pd.DataFrame()}
        self.test_psds_buffer_teacher = {self.threshold: pd.DataFrame()}

    def test_step(self, batch, batch_idx):
        audio, labels, padded_indxs, filenames = batch

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher)

        # compute psds
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            weak_preds_student,
            filenames,
            self.encoder,
            thresholds=list(self.test_psds_buffer_student.keys()),
            median_filter=self.hparams["training"]["median_window"],
            decode_weak=self.decode_weak_mode,
        )

        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = self.test_psds_buffer_student[th].append(
                decoded_student_strong[th], ignore_index=True
            )

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            weak_preds_teacher,
            filenames,
            self.encoder,
            thresholds=list(self.test_psds_buffer_teacher.keys()),
            median_filter=self.hparams["training"]["median_window"],
            decode_weak=self.decode_weak_mode,
        )

        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[th].append(
                decoded_teacher_strong[th], ignore_index=True
            )

    def on_test_epoch_end(self):
        # calculate the metrics
        psds_score_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer_student,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
        )

        psds_score_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer_student,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )

        psds_score_teacher_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
        )

        psds_score_teacher_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )

        self.result_dict = {
            "threshold": self.threshold,
            "student/PSDS1": psds_score_scenario1,
            "student/PSDS2": psds_score_scenario2,
            "teacher/PSDS1": psds_score_teacher_scenario1,
            "teacher/PSDS2": psds_score_teacher_scenario2,
        }


def main(config, ckpt_path: str):
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    sed_student = getattr(models, config["net"]["name"])(**config["net"])
    mode = TEST
    test_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    test_dataset = StronglyAnnotatedSet(
        config["data"]["test_folder"],
        test_df,
        encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )

    desed_training = SEDTask4Custom(config, encoder=encoder, sed_student=sed_student, mode=mode, test_data=test_dataset)
    desed_training.load_state_dict_by_ckpt_path(ckpt_path)
    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        gpus=1,
        strategy=config["training"]["backend"],
        limit_test_batches=1.0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=40,
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=100,
    )

    result_list = []
    thres_num = config["training"]["n_test_thresholds"]
    threshold_list = np.arange(1 / (thres_num * 2), 1, 1 / thres_num)
    with open("./psds_per_thres.txt", "w") as f:
        for thres in threshold_list:
            print(f"THRESHOLD {thres} TESTING...")
            desed_training.set_threshold(thres)
            trainer.test(desed_training)

            result = desed_training.result_dict
            f.write(
                f"{result['threshold']} {result['student/PSDS1']} {result['student/PSDS2']} {result['teacher/PSDS1']} {result['teacher/PSDS2']}\n"
            )
            print(result)
            result_list.append(result)

    df = pd.DataFrame(result_list)
    df.sort_values("student/PSDS1", inplace=True, ascending=True)
    student_pds1_best_series = df.iloc[-1]
    df.sort_values("student/PSDS2", inplace=True, ascending=True)
    student_pds2_best_series = df.iloc[-1]
    df.sort_values("teacher/PSDS1", inplace=True, ascending=True)
    teacher_pds1_best_series = df.iloc[-1]
    df.sort_values("teacher/PSDS2", inplace=True, ascending=True)
    teacher_pds2_best_series = df.iloc[-1]

    with open("./best_thres_per_psds.txt", "w") as f:
        f.write(f"{student_pds1_best_series['student/PSDS1']} {student_pds1_best_series['threshold']}\n")
        f.write(f"{student_pds2_best_series['student/PSDS2']} {student_pds2_best_series['threshold']}\n")
        f.write(f"{teacher_pds1_best_series['teacher/PSDS1']} {teacher_pds1_best_series['threshold']}\n")
        f.write(f"{teacher_pds2_best_series['teacher/PSDS2']} {teacher_pds2_best_series['threshold']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--conf_file", default=None)
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.ckpt_path)
