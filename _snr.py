
def signal_noise_ratio(
    x: Tensor,
    y: Tensor,
    frame_size: int,
    hop_size: int,
    seek_size: int,
    max_clamp: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """
    Calculate framewise SNR with time-domain linear-phase correction.
    This function searches the maximum possible SNR.
    TODO: Optimize the implementation.
    Args:
        x, y: [..., n_sample].
        frame_size: SNR frame size.
        hop_size: shift of frame in number of samples.
        seek_size: minimum linear-phase search region.
        max_clamp: SNR can be up to infinity.
                   Clamp values above the given value.

    Returns:
        max_snr: [..., n_frame] the best SNR in each frame.
        idx: [..., n_frame] the location of the best SNR.
    """
    seek = seek_size * 2 + 1
    batch_size = x.shape[:-1]
    # prod = prod(batch_size)
    n_sample = x.size(-1)
    x = x.reshape(-1, 1, n_sample)
    y = y.reshape(-1, 1, n_sample)
    # [prod, 1, n_sample]

    # processing x:
    x = frame_signal(x, frame_size, hop_size)
    # [prod, frame_size, n_frame]
    xx = (x * x).unsqueeze(1).repeat(1, seek, 1, 1)
    x = x.unsqueeze(1).repeat(1, seek, 1, 1)
    # [prod, seek, frame_size, n_frame]

    # processing y:
    y = pad(y, [seek_size, seek_size])
    # [prod, 1, n_sample + seek - 1]
    y = frame_signal(y, n_sample, 1).permute(0, 2, 1)
    # [prod, seek, n_sample]
    y = y.reshape(-1, 1, n_sample)
    # [prod* seek, 1, n_sample]
    y = frame_signal(y, frame_size, hop_size)
    # [prod* seek, frame_size, n_frame]
    y = y.reshape(x.size(0), seek, frame_size, -1)
    # [prod, seek, frame_size, n_frame]

    # compute SNR:
    all_snr = (xx.sum(-2) / ((x - y) ** 2).sum(-2))
    all_snr = 10 * torch.log10(all_snr)
    # [prod, seek, n_frame]
    max_snr, idx = all_snr.max(-2)
    # [prod, n_frame]
    max_snr = max_snr.reshape(*batch_size, -1)
    idx = idx.reshape(*batch_size, -1)
    # [..., n_frame]

    # max clamping:
    if max_clamp:
        max_snr = max_snr.clamp(max=max_clamp)
    return max_snr, idx
