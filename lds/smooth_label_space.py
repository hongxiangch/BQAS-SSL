from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
import numpy as np


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_bin_idx(label, args):
    if args.original_label:
        min_label = -8.0
        max_label = 6.1
    else:
        min_label = 0.0
        max_label = 1.0
    bins = args.bins
    step = (max_label-min_label)/bins
    index = 0
    for b in range(1, bins+1):
        if label <= min_label+(step*b):
            index = b - 1
            break

    return index


def smooth(labels, args):
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
    bin_index_per_label = [get_bin_idx(label, args) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = args.bins
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel=args.kernel, ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    return eff_label_dist, bin_index_per_label
