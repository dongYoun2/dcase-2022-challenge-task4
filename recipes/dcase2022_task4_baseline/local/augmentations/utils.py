import torch


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)
