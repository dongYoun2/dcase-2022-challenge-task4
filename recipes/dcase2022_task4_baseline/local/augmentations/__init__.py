import torch
from torch import nn
import numpy as np
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.dataio.dataloader import make_dataloader

from local.augmentations.utils import compute_amplitude, dB_to_amplitude


def inner_filter_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    if type(filter_type) is not str:
        raise TypeError("filter_type has to be type string")

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()  # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1

        band_bndry_freqs = (
            torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1, (n_freq_band - 1,)))[0]
            + torch.arange(1, n_freq_band) * min_bw
        )
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = (
                torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            )
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = (
                    band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
                )

        elif filter_type == "linear":
            band_factors = (
                torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            )
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)

            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = torch.linspace(
                        band_factors[j, i], band_factors[j, i + 1], band_bndry_freqs[i + 1] - band_bndry_freqs[i]
                    ).unsqueeze(-1)

            freq_filt = 10 ** (freq_filt / 20)

        return features * freq_filt
    else:
        return features


def filter_aug(features, n_transform, filter_db_range, filter_bands, filter_minimum_bandwidth, filter_type):
    if n_transform not in [0, 1, 2]:
        raise Exception("n_transform has to be 0, 1 or 2")

    feature_tuple = tuple(
        inner_filter_aug(
            features,
            db_range=filter_db_range,
            n_band=filter_bands,
            min_bw=filter_minimum_bandwidth,
            filter_type=filter_type,
        )
        for _ in range(n_transform)
    )
    if len(feature_tuple) == 0:
        return features, features
    elif len(feature_tuple) == 1:
        return feature_tuple[0], feature_tuple[0]
    else:
        return feature_tuple


def time_mask(features, labels=None, mask_ratios: float = 0.1):
    """Mask to zeros some frames of the features and labels.

    Args:
        features (torch.Tensor): Data.
        labels (torch.Tensor, optional): Labels. Defaults to None.
        mask_ratios (float, optional): Masking ratio. Defaults to 0.1.

    Returns:
        (masked_features) or (masked_features, masked_labels)
    """
    features = features.transpose(0, -1)
    f_frames = len(features)
    f_width = np.random.randint(0, int(f_frames * mask_ratios))
    f_low = np.random.randint(0, f_frames - f_width)
    features[f_low : f_low + f_width] = 0
    if labels is None:
        return features.transpose(0, -1)
    else:
        _, _, l_frames = labels.shape
        l_width = round((f_width / f_frames) * l_frames)
        l_low = round((f_low / f_frames) * l_frames)
        labels[:, :, l_low : l_low + l_width] = 0
        return features.transpose(0, -1), labels


def frame_shift(data, labels, shift_rate=0.2):
    batch_size = len(data)
    shift_rates = np.random.uniform(-shift_rate, shift_rate, batch_size)
    data_len = data.shape[1]
    for i, shift in enumerate(data_len * shift_rates):
        shift = int(round(shift))
        data[i] = torch.roll(data[i], shift, dims=0)
    labels_len = labels.shape[2]
    for i, shift in enumerate(labels_len * shift_rates):
        shift = int(round(shift))
        labels[i] = torch.roll(labels[i], shift, dims=1)
    return data, labels


class SpecAugment(nn.Module):
    """SpecAugment
    ref: https://github.com/TaoRuijie/ECAPA-TDNN/blob/c9d6ba065bfb7c6b7a17b1b062a7e79314bce411/model.py#L96-L130

    Args:
        freq_mask_width (tuple, optional): Size of frequency mask. Defaults to (0, 8).
        time_mask_width (tuple, optional): Size of time mask. Defaults to (0, 10).
    """

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        if self.training:
            if self.freq_mask_width[1] > 0:
                x = self.mask_along_axis(x, dim=1)
            if self.time_mask_width[1] > 0:
                x = self.mask_along_axis(x, dim=2)
        return x


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    csv_keys : list, None, optional
        Default: None . One data entry for the noise data should be specified.
        If None, the csv file is expected to have only one data entry.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.

    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> noisifier = AddNoise('samples/noise_samples/noise.csv')
    >>> noisy = noisifier(clean, torch.ones(1))
    """

    def __init__(
        self,
        csv_file="/nfs/train/desed-lab/data/musan_music.csv",
        csv_keys=None,
        sorting="random",
        num_workers=0,
        snr_low=5,
        snr_high=20,
        pad_noise=False,
        mix_prob=1.0,
        start_index=None,
        normalize=False,
        replacements={},
    ):
        super().__init__()

        self.csv_file = csv_file
        self.csv_keys = csv_keys
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.replacements = replacements

    def forward(self, waveforms, lengths=None):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        if lengths is None:
            lengths = torch.ones(waveforms.shape[0]).to(waveforms)
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(waveforms)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length = self._load_noise(
                lengths,
                tensor_length,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
            noisy_waveform += noise_waveform

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = torch.max(torch.abs(noisy_waveform), dim=1, keepdim=True)
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        lengths = lengths.long().squeeze(1)
        batch_size = len(lengths)

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = ExtendedCSVDataset(
                    csvpath=self.csv_file,
                    output_keys=self.csv_keys,
                    sorting=self.sorting if self.sorting != "random" else "original",
                    replacements=self.replacements,
                )
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=(self.sorting == "random"),
                )
                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                min_len = torch.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            padding = (0, max_length - noise_batch.size(1))
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(high=max_chop, size=(1,), device=lengths.device)

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index : start_index + max_length]
        noise_len = (noise_len - start_index).clamp(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(noise_batch, noise_lens, added_noise, added_lens)

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    @staticmethod
    def _concat_batch(noise_batch, noise_lens, added_noise, added_lens):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]
        pad = (0, abs(noise_tensor_len - added_tensor_len))
        if noise_tensor_len > added_tensor_len:
            added_noise = torch.nn.functional.pad(added_noise, pad)
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = torch.nn.functional.pad(noise_batch, pad)
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len

        noise_batch = torch.cat((noise_batch, added_noise))
        noise_lens = torch.cat((noise_lens, added_lens))

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            noises, lens = next(self.noise_data).at_position(0)
        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens = next(self.noise_data).at_position(0)
        return noises, lens


if __name__ == "__main__":
    add_music = AddNoise()
    waveforms = torch.randn(4, 160000)
    noisy_waveforms = add_music(waveforms)
    print(noisy_waveforms.shape)
