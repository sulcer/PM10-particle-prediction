import numpy as np


def create_time_series(data, window_size, feature_cols):
    sequences = []
    n_samples = len(data)

    for i in range(window_size, n_samples + 1):
        sequence = data[i - window_size:i, feature_cols]
        sequences.append(sequence)

    return np.array(sequences)
