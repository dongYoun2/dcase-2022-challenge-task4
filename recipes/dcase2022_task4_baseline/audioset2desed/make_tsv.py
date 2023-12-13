from glob import glob
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from audioset2desed.classes_dict import mapping

AUDIOSET_ROOT = "db/audioset"


def main(split="train"):
    ann = pd.read_csv(f"{AUDIOSET_ROOT}/audioset_{split}_strong.tsv", sep="\t")
    # create dict for machine ID
    mid2name = pd.read_csv(f"{AUDIOSET_ROOT}/mid_to_display_name.tsv", sep="\t", names=["mid", "name"])
    mid_dict = dict()
    for mid, name in mid2name.itertuples(index=False):
        mid_dict[mid] = name
    # create dict for audioset2desed label
    audioset2desed = dict()
    for desed_l, audioset_l_list in mapping.items():
        for audioset_l in audioset_l_list:
            audioset2desed[audioset_l] = desed_l
    # create available audio list
    audios = [path.split("/")[-1][:-4] for path in glob(f"{AUDIOSET_ROOT}/train_strong/*.wav")]
    # create audioset2desed annotation
    ret = []
    for filename, onset, offset, event_label in tqdm(ann.itertuples(index=False), total=len(ann)):
        event_label = mid_dict[event_label]
        if filename in audios:  # and event_label in audioset2desed:
            ret.append((f"{filename}.wav", onset, offset, event_label))
    return pd.DataFrame(ret, columns=["filename", "onset", "offset", "event_label"])


if __name__ == "__main__":
    df = main("train")
    df.to_csv(f"{AUDIOSET_ROOT}/audioset2desed_train_strong.tsv", sep="\t", index=False)
