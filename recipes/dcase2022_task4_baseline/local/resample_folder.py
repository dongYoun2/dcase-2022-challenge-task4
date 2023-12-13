import argparse
import glob
import os
from pathlib import Path
import multiprocessing as mp
from itertools import repeat

import librosa
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

parser = argparse.ArgumentParser("Resample a folder recursively")
parser.add_argument(
    "--in_dir",
    type=str,
    default="/media/sam/bx500/DCASE_DATA/dataset/audio/validation/",
)
parser.add_argument("--out_dir", type=str, default="/tmp/val16k")
parser.add_argument("--target_fs", default=16000)
parser.add_argument("--regex", type=str, default="*.wav")


def resample(audio, orig_fs, target_fs):
    """
    Resamples the audio given as input at the target_fs sample rate, if the target sample rate and the
    original sample rate are different.

    Args:
        audio (Tensor): audio to resample
        orig_fs (int): original sample rate
        target_fs (int): target sample rate

    Returns:
        Tensor: audio resampled
    """
    out = []
    for c in range(audio.shape[0]):
        tmp = audio[c].detach().cpu().numpy()
        if target_fs != orig_fs:
            tmp = librosa.resample(tmp, orig_fs, target_fs)
        out.append(torch.from_numpy(tmp))
    out = torch.stack(out)
    return out


def audio_prep(x):
    f, target_fs, in_dir, out_dir = x
    try:
        audio, _ = librosa.load(f, sr=target_fs, mono=True, duration=10)
        os.makedirs(
            Path(os.path.join(out_dir, Path(f).relative_to(Path(in_dir)))).parent,
            exist_ok=True,
        )
        sf.write(os.path.join(out_dir, Path(f).relative_to(Path(in_dir))), audio, target_fs)
    except Exception as e:
        print(f"Error loading {f} for {e}")


def resample_folder(in_dir, out_dir, target_fs=16000, regex="*.wav", use_mp=False):
    """
    Resamples the audio files contained in the in_dir folder and saves them in out_dir folder

    Args:
        in_dir (str): path to audio directory (audio to be resampled)
        out_dir (str): path to audio resampled directory
        target_fs (int, optional): target sample rate. Defaults to 16000.
        regex (str, optional): regular expression for extension of file. Defaults to "*.wav".
    """
    compute = True
    files = glob.glob(os.path.join(in_dir, regex))
    if os.path.exists(out_dir):
        out_files = glob.glob(os.path.join(out_dir, regex))
        if len(files) == len(out_files):
            compute = False

    if compute and use_mp:
        with mp.Pool(mp.cpu_count() // 2) as p, tqdm(total=len(files), desc=f"{in_dir[:30]}...") as pbar:
            for _ in tqdm(p.imap_unordered(audio_prep, zip(files, repeat(target_fs), repeat(in_dir), repeat(out_dir)))):
                pbar.update()
    elif compute and not use_mp:
        with tqdm(total=len(files), desc="") as pbar:
            for _ in tqdm(map(audio_prep, zip(files, repeat(target_fs), repeat(in_dir), repeat(out_dir)))):
                pbar.update()
    return compute


if __name__ == "__main__":
    args = parser.parse_args()
    resample_folder(args.in_dir, args.out_dir, int(args.target_fs), args.regex)
