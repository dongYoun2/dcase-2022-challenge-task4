import argparse
import os
from glob import glob
import subprocess
import multiprocessing as mp

from tqdm import tqdm


def subproc(x):
    conf_path, ckpt, args = x
    subprocess.run(
        f"""
        PYTHONPATH=. python train_sed.py --conf_file {conf_path} --eval_from_checkpoint {ckpt} --temperature {args.temperature}
        """,
        shell=True,
    )


def main(args):
    print("Checking for ckpt files...")
    conf_path = f"{args.ckpt_dir}/hparams.yaml"
    ckpt_list = glob(f"{args.ckpt_dir}/epoch=*.ckpt")
    ckpt_epoch_range = list(range(*args.ckpt_range))

    work_list = []

    for ckpt in ckpt_list:
        epoch = int(ckpt.split("/")[-1][6:].split("-")[0])
        if epoch in ckpt_epoch_range and not os.path.exists(
            f"{args.ckpt_dir}/metrics_eval_eval21_{ckpt.split('/')[-1][:-5]}/scores.npy"
        ):
            work_list.append((conf_path, ckpt, args))

    with mp.Pool(processes=5) as pool, tqdm(total=len(work_list)) as pbar:
        for _ in pool.imap_unordered(subproc, work_list):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--ckpt_range", nargs="+", type=int, default=[349, 9999])
    parser.add_argument("--temperature", default=3)

    _args = parser.parse_args()

    main(_args)
