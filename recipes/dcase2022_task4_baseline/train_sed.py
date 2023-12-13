import argparse
import time
import warnings

import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

from pathlib import Path
from typing import Optional, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet, WeakSet
from desed_task.utils.encoder import ManyHotEncoder

from local.classes_dict import classes_labels
from local.sed_trainer import SEDTask4
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations
from local.modes import TRAIN, TEST, EVALUATION
from local import schedulers, optimizers, models

warnings.simplefilter(action="ignore", category=FutureWarning)


def resample_data_generate_durations(config_data, mode):
    if mode == TRAIN:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "audioset_strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    elif mode == TEST:
        dsets = ["test_folder"]
    elif mode == EVALUATION:
        dsets = ["eval_folder"]
    else:
        raise Exception("mode has to be TRAIN or TEST or EVALUATION")

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"], use_mp=True
        )

    if mode != EVALUATION:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(config_data[base_set + "_folder"], config_data[base_set + "_dur"])


def single_run(
    config,
    exp_dir,
    gpus,
    mode,
    ckpt_path=None,
    strong_real=False,
    train_more_synth=False,
    train_audioset2desed=False,
    train_validation=False,
    fast_dev_run=False,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        exp_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"exp_dir": exp_dir})

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if mode != EVALUATION:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )
    else:
        devtest_dataset = UnlabeledSet(config["data"]["eval_folder"], encoder, pad_to=None, return_filename=True)

    test_dataset = devtest_dataset

    ##### model definition  ############
    model_name = config["net"].get("name", "CRNN")
    sed_student = getattr(models, model_name)(**config["net"])

    if mode == TRAIN:
        ##### data prep train valid ##########
        strong_set_list = []
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )
        strong_set_list.append(synth_set)

        if strong_real:
            strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
            strong_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                strong_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )
            strong_set_list.append(strong_set)

        if train_more_synth:
            synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
            synth_val = StronglyAnnotatedSet(
                config["data"]["synth_val_folder"],
                synth_df_val,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )
            strong_set_list.append(synth_val)

        if train_audioset2desed:
            audioset_df = pd.read_csv(config["data"]["audioset_strong_tsv"], sep="\t")
            audioset = StronglyAnnotatedSet(
                config["data"]["audioset_strong_folder"],
                audioset_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )
            strong_set_list.append(audioset)

        if train_validation:
            validation_df = pd.read_csv(config["data"]["validation_tsv"], sep="\t")
            validationset = StronglyAnnotatedSet(
                config["data"]["validation_folder"],
                validation_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )
            strong_set_list.append(validationset)

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        real_df_val = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        real_val = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            real_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )

        if len(strong_set_list) > 1:
            strong_full_set = torch.utils.data.ConcatDataset(strong_set_list)
            tot_train_data = [strong_full_set, weak_set, unlabeled_set]
        else:
            tot_train_data = [synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset([real_val, weak_val])

        ##### training params and optimizers ############
        step_cnt_per_epoch = min(
            [
                len(tot_train_data[indx])
                // (config["training"]["batch_size"][indx] * config["training"]["accumulate_batches"])
                for indx in range(len(tot_train_data))
            ]
        )

        opt = getattr(optimizers, config["opt"]["name"])(sed_student.parameters(), **config["opt"]["params"])
        exp_scheduler = {
            "scheduler": getattr(schedulers, config["scheduler"]["name"])(
                opt,
                steps_per_epoch=step_cnt_per_epoch,
                n_epochs=config["training"]["n_epochs"],
                **config["scheduler"]["params"],
            ),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["exp_dir"]),
            config["exp_dir"].split("/")[-1],
        )
        print(f"experiment version dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                **config["training"]["checkpoint_params"],
            ),
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = False
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        mode=mode,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        config["training"]["n_epochs"] = 3
        config["training"]["validation_interval"] = 1

    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    if len(gpus.split(",")) > 1:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=config["training"]["n_epochs"],
        callbacks=callbacks,
        gpus=gpus,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=ckpt_path if mode == TRAIN else None,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    if mode == TRAIN:
        # start tracking energy consumption
        trainer.fit(desed_training)
        ckpt_path = trainer.checkpoint_callback.best_model_path

    desed_training.load_state_dict_by_ckpt_path(ckpt_path)
    trainer.test(desed_training)


def set_seed(seed: int):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)


def get_default_exp_name():
    assert time.strftime("%Z", time.localtime(time.time())) == "KST"
    exp_name = time.strftime("%m%d-%H%M%S", time.localtime(time.time()))

    return exp_name


def get_exp_name_from_ckpt_path(ckpt_path: str):
    # assume checkpoint path is in exp version directory
    ckpt_abs_path: str = str(Path(ckpt_path).resolve())
    exp_name = ckpt_abs_path.split("/")[-3]

    return exp_name


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    # for training conf and logging
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--save_path",
        default="desed-lab/exps",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--exp_name",
        help="Experiment name (goes after the `save_path`)",
    )
    # more training sets
    parser.add_argument(
        "--strong_real",
        type=str2bool,
        default=True,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--train_more_synth",
        type=str2bool,
        default=True,
        help="valid set이었던 synthetic21_validation을 훈련 데이터로 추가합니다",
    )
    parser.add_argument(
        "--train_audioset2desed",
        action="store_true",
        default=False,
        help="DESED label로 mapping 된 AudioSet (strong)을 훈련 데이터로 활용합니다",
    )
    parser.add_argument(
        "--train_validation",
        action="store_true",
        default=False,
        help="(제출용) validation set을 훈련 데이터로 추가합니다. test 데이터가 `public_eval`로 변경되고, valid 시점마다 checkpoint 생성됨",
    )
    # for checkpoint paths
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument("--test_from_checkpoint", default=None, help="Test the model specified")
    parser.add_argument("--eval_from_checkpoint", default=None, help="Evaluate the model specified")
    # misc
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', " "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument("--resample", action="store_true", help="Resample to 16kHz")
    parser.add_argument(
        "--temperature",
        help="Temperature for the sigmoid at inference. Edit `config.yaml` for training",
    )

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    if args.test_from_checkpoint is None and args.eval_from_checkpoint is None:
        mode = TRAIN
        ckpt_path: Optional[str] = args.resume_from_checkpoint
    elif args.test_from_checkpoint is not None and args.eval_from_checkpoint is None:
        mode = TEST
        ckpt_path: str = args.test_from_checkpoint
    elif args.test_from_checkpoint is None and args.eval_from_checkpoint is not None:
        mode = EVALUATION
        ckpt_path: str = args.eval_from_checkpoint
    else:
        raise Exception("have to use either test_from_checkpoint or eval_from_checkpoint")

    if args.exp_name is None and ckpt_path is None:
        exp_name: str = get_default_exp_name()
    elif args.exp_name is None and ckpt_path is not None:
        exp_name: str = get_exp_name_from_ckpt_path(ckpt_path)
    else:
        exp_name: str = args.exp_name

    if mode == EVALUATION:
        config["training"]["batch_size_val"] = 1

    seed = config["training"]["seed"]
    if seed:
        set_seed(seed)

    if args.resample:
        resample_data_generate_durations(config["data"], mode)

    if args.train_validation:
        config["data"]["validation_folder"] = "../../data/dcase/dataset/audio/validation/validation_16k/"
        config["data"]["validation_folder_44"] = "../../data/dcase/dataset/audio/validation/validation/"
        config["data"]["validation_tsv"] = "../../data/dcase/dataset/metadata/validation/validation.tsv"
        config["data"]["test_folder"] = "../../data/DESED_public_eval/audio/eval/public_16k/"
        config["data"]["test_folder_44k"] = "../../data/DESED_public_eval/audio/eval/public/"
        config["data"]["test_tsv"] = "../../data/DESED_public_eval/metadata/eval/public.tsv"
        config["data"]["test_dur"] = "../../data/DESED_public_eval/metadata/eval/public_durations.tsv"
        config["training"]["checkpoint_params"] = {
            "every_n_epochs": config["training"]["validation_interval"],
            "save_last": True,
            "save_top_k": -1,
        }

    if args.temperature is not None:
        config["net"]["T"] = args.temperature

    single_run(
        config,
        f"{args.save_path}/{exp_name}",
        args.gpus,
        mode,
        ckpt_path,
        args.strong_real,
        args.train_more_synth,
        args.train_audioset2desed,
        args.train_validation,
        args.fast_dev_run,
    )
