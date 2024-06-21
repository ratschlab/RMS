import itertools
import fractions
import multiprocessing
import contextlib

import math
import numba
import numpy as np

import scipy as sp
import scipy.interpolate as sp_ip
import scipy.cluster.vq as sq_clust_vq
import scipy.signal as sp_signal
import scipy.sparse as sp_sparse
import scipy.signal._arraytools as sp_sig_arraytools
import scipy.stats as sp_stats

import bottleneck as bn

@numba.jit(nogil=True)
def in_range(x, y, max_dist):
    """Returns True iff a vector is in the range of another vector

    :param x: The first vector
    :param y: The second vector
    :param max_dist: The maximum distance the vectors are allowed to differ in each element

    """
    assert x.ndim == 1 and x.shape == y.shape

    for i in range(x.shape[0]):
        if abs(x[i] - y[i]) > max_dist:
            return False

    return True

@numba.jit(nogil=True)
def embed_seq(x, tau, d):
    """Builds an embedding sequence of the time series

    :param x: The original time series
    :param tau: The embedding time lag, i.e. how many values to skip
    :param d: The maximum embedding dimension, i.e. the length of the embeddings
    """
    n = x.shape[0]

    assert d * tau <= n, "Cannot build such a matrix, because d*tau>n"
    assert tau > 0, "tau has to be at least 1"

    num_embeds = n - (d - 1) * tau
    embeds = np.empty(shape=(num_embeds, d), dtype=x.dtype)

    for i in range(num_embeds):
        embed = x[i:(i + d * tau):tau]

        for j in range(d):
            embeds[i, j] = embed[j]

    return embeds

def filtfilt(b, a, x, zi=None):
    """
    A forward-backward filter.
    This function applies a linear filter twice, once forward and once
    backwards.  The combined filter has linear phase.
    The function provides options for handling the edges of the signal.
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
    Returns
    -------
    y : ndarray
        The filtered output, an array of type numpy.float64 with the same
        shape as `x`.
    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    x = np.asarray(x)

    axis = -1
    ntaps = max(len(a), len(b))
    edge = ntaps * 3

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be at least "
                         "padlen, which is %d." % edge)

    # Make an extension of length `edge` at each
    # end of the input array.
    ext = sp_sig_arraytools.odd_ext(x, edge, axis=axis)

    # Get the steady state of the filter's step response.
    if zi is None:
        zi = sp_signal.lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = sp_sig_arraytools.axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = sp_signal.lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = sp_sig_arraytools.axis_slice(y, start=-1, axis=axis)
    (y, zf) = sp_signal.lfilter(b, a, sp_sig_arraytools.axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = sp_sig_arraytools.axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = sp_sig_arraytools.axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y



def create_filter(name, cutoff, fs, pass_zero, **kwargs):
    nyq_freq = .5 * fs
    is_band_pass = not np.isscalar(cutoff) and (not pass_zero)
    is_band_stop = not np.isscalar(cutoff) and pass_zero
    is_low_pass = np.isscalar(cutoff) and pass_zero
    is_high_pass = np.isscalar(cutoff) and (not pass_zero)

    if name.lower() == 'kaiser':
        width = kwargs.get('width') or 0.5 / nyq_freq
        ripple_db = kwargs.get('ripple_db') or 60.0
        order, beta = sp_signal.kaiserord(ripple_db, width)

        taps = sp_signal.firwin(order, cutoff, window=('kaiser', beta),
                             pass_zero=pass_zero, nyq=nyq_freq)

        return taps, np.ones(1, dtype=float)
    elif name.lower() == 'butter':
        order = kwargs.get('order') or 7

        if np.isscalar(cutoff):
            cutoff /= nyq_freq
        else:
            cutoff = [c / nyq_freq for c in cutoff]

        if is_high_pass:
            btype = 'highpass'
        elif is_low_pass:
            btype = 'lowpass'
        elif is_band_pass:
            btype = 'bandpass'
        else:
            assert is_band_stop
            btype = 'bandstop'

        b, a = sp_signal.butter(order, cutoff, btype=btype, analog=False, output='ba')

        return b, a
    else:
        order = kwargs.get('order') or 7
        taps = sp_signal.firwin(order, cutoff, window=name,
                             pass_zero=pass_zero, nyq=nyq_freq)

        return taps, np.ones(1, dtype=float)

## Differenced time series
def differenced(x):
    return np.diff(x)

## Mark a time series invalid if it has too few observed values
def mark_invalid(raw_signal,threshold=0.5):
    if np.mean(np.isnan(raw_signal)) >= self._threshold:
        return np.nan
    else:
        return raw_signal

## Log-transform of time series
def log_tf(x):
    return np.log(x)

## One hot encoding of input
def one_hot_encode(x,n_classes):
    ZERO = np.zeros(shape=1, dtype=np.int64)
    TRUE = np.ones(shape=1, dtype=np.bool_)
    
    if len(x) != 1:
        raise ValueError("Can only one-hot encode single number but was given: {}".format(len(x)))

    val = x[0]

    if int(val) != val:
        raise ValueError("Can only on-hot encode integers, but was given: {}".format(val))

    val = int(val)

    if not 0 <= val < self.n_classes:
        raise ValueError("Value {} outside of class range [0, {})".format(val, self.n_classes))

    matrix = sp_sparse.csr_matrix((self.TRUE, (self.ZERO, np.atleast_1d(val))))
    return matrix

## Returns the array of indices which are relative extrema and also greater than or equal than
#  some percentile of the input data
def perc_rel_maxima(raw_signal,perc):
    relmax = signal.argrelmax(raw_signal)[0]
    perc_score = np.percentile(raw_signal, perc)
    percrelmax = relmax[raw_signal[relmax] >= perc_score]
    return np.array(percrelmax)

## Number of samples in time-span
def time_to_samples(seconds, sampling_freq):
    return int(seconds * sampling_freq)

## Apply a moving window function to a NP array
def moving_window(func, arr, nperseg, noverlap=0, axis=-1):
    """

    :param func: The function to apply to every window
    :param arr: The array over which the window moves
    :param nperseg: The size of the window
    :param noverlap: How much consecutive windows overlap
    :param axis: Along which axis the window moves
    """
    arr_win = window_reshape(arr, nperseg=nperseg, noverlap=noverlap, axis=axis)

    # window_reshape will roll `axis` to the last position, therefore
    # the windows will be at the last position and the 'original' axis
    # would be at the second last position. Thus we
    # apply the function to the last axis.
    result = np.apply_along_axis(
        func,
        axis=-1,
        arr=arr_win
    )
    # And here we need to undo the roll done by window_reshape because
    # the last dimension will hold the result of the window computations.
    result = np.rollaxis(result, axis, -1)
    return np.asarray(result)

def moving_windows(func, arrs, nperseg, noverlap=0, axis=-1):
    """

    :param func: The function to apply to every window, receives as many arguments as there are arrays
    :param arrs: The arrays over which the window moves in parallel
    :param nperseg: The size of the window
    :param noverlap: How much consecutive windows overlap
    :param axis: Along which axis the window moves
    :return:
    """
    shape = arrs[0].shape
    nperseg = int(nperseg)
    noverlap = int(noverlap)

    if noverlap >= nperseg:
        raise ValueError("noverlap={} must be smaller than nperseg={}".format(noverlap, nperseg))

    if not all(shape == arr.shape for arr in arrs):
        raise ValueError("not all arrays have the same shape {}".format(shape))

    def reshape(arr):
        arr = window_reshape(arr, nperseg=nperseg, noverlap=noverlap, axis=axis)
        return arr

    arr_wins = list(map(reshape, arrs))

    # window_reshape will roll `axis` to the last position, therefore
    # the windows will be at the last position and the 'original' axis
    # would be at the second last position. Thus we
    # apply the function to the last axis.
    result = apply_along_axis_n(func, axis=-1, arrs=arr_wins)
    # And here we need to undo the roll done by window_reshape because
    # the last dimension will hold the result of the window computations.
    result = np.rollaxis(result, axis, -1)

    return np.asarray(result)

## Moving window sampling rate
def moving_window_fs(source_fs, nperseg, noverlap):
    step = nperseg - noverlap
    return source_fs / step

@numba.jit(nogil=True)
def subarray_min(x, beg, end):
    if beg.shape[0] != end.shape[0]:
        raise ValueError("beg and end must have the same length")

    x_len = x.shape[0]
    mins = np.empty(beg.shape[0], dtype=x.dtype)

    for i in range(beg.shape[0]):
        mins[i] = np.min(x[max(0, beg[i]):min(end[i], x_len)])

    return mins

@numba.jit(nogil=True)
def subarray_max(x, beg, end):
    if beg.shape[0] != end.shape[0]:
        raise ValueError("beg and end must have the same length")

    x_len = x.shape[0]
    maxs = np.empty(beg.shape[0], dtype=x.dtype)

    for i in range(beg.shape[0]):
        maxs[i] = np.max(x[max(0, beg[i]):min(end[i], x_len)])

    return maxs

## Returns the indices of all relative minima
@numba.jit(nogil=True)
def simple_argrelmin(x):
    assert(x.ndim == 1)
    n = x.shape[0]
    num_min = 0
    diff = x[1] - x[0]
    pos_prev = diff > 0.
    neg_prev = diff < 0.
    saddle_prev = not (pos_prev or neg_prev)
    saddle_start = 0
    locations = np.empty(n // 3, dtype=np.int32)

    for i in range(1, n - 1):
        diff = x[i + 1] - x[i]
        pos = diff > 0.
        neg = diff < 0.

        if pos and neg_prev:
            if saddle_prev:
                locations[num_min] = saddle_start
                num_min += 1
            else:
                locations[num_min] = i
                num_min += 1

        if pos or neg:
            pos_prev = pos
            neg_prev = neg
            saddle_prev = False
        elif not saddle_prev:
            saddle_prev = True
            saddle_start = i
        else:
            saddle_prev = True

    return locations[:num_min]

## Returns the indices of all relative maxima
@numba.jit(nogil=True)
def simple_argrelmax(x):
    assert(x.ndim == 1)
    n = x.shape[0]
    num_min = 0
    diff = x[1] - x[0]
    pos_prev = diff > 0.
    neg_prev = diff < 0.
    saddle_prev = not (pos_prev or neg_prev)
    saddle_start = 0
    locations = np.empty(n // 3, dtype=np.int32)

    for i in range(1, n - 1):
        diff = x[i + 1] - x[i]
        pos = diff > 0.
        neg = diff < 0.

        if neg and pos_prev:
            if saddle_prev:
                locations[num_min] = saddle_start
                num_min += 1
            else:
                locations[num_min] = i
                num_min += 1

        if pos or neg:
            pos_prev = pos
            neg_prev = neg
            saddle_prev = False
        elif not saddle_prev:
            saddle_prev = True
            saddle_start = i
        else:
            saddle_prev = True

    return locations[:num_min]

## Returns a slice from the input
def slice_column(x,start,stop=None,step=None):
    start = 0 if stop is None else start
    stop = start if stop is None else stop
    step = step or 1
    return x.toarray()[:, start:stop:step]

def encode_2d(x, y, x_clip, y_clip, n_rows, n_cols, normed=True):
    assert(x_clip > 0)
    assert(y_clip > 0)
    assert(n_rows > 0)
    assert(n_cols > 0)
    x_clipped = np.clip(x - np.mean(x), -x_clip, x_clip)
    y_clipped = np.clip(y - np.mean(y), -y_clip, y_clip)
    x_bins = np.linspace(-x_clip, x_clip, n_cols+1, endpoint=True)
    y_bins = np.linspace(-y_clip, y_clip, n_rows+1, endpoint=True)
    grid, _, _ = np.histogram2d(x_clipped, y_clipped, (x_bins, y_bins), normed=normed)
    grid.shape = (-1,)
    return grid

def trace(x,y,x_clip,y_clip,n_bins):
    x_clip = abs(x_clip)
    y_clip = abs(y_clip)
    n_bins = int(n_bins)
    
    if n_bins < 1:
        raise ValueError("Number of bins must be at least 1")

    return encode_2d(x, y, self.x_clip, self.y_clip, self.n_bins, self.n_bins, True)

## Returns FALSE if the signal is invalid and otherwise pass signal through
def signal_valid(sig,constant_std_ts=0.001,nan_ratio_discard=0.5,check_std=True):
    if np.sum(np.isnan(sig)) < nan_ratio_discard * sig.shape[0]: 
        if not check_std or np.nanstd(sig) > constant_std_ts:
            return sig
    else:
        return np.nan

## Marks inprobable samples of a signal invalid with a symbolic value
def mark_invalid_low_high_threshold(sig,low,high):
    new_sig = np.copy(sig)
    err_set = np.geterr()
    np.seterr(invalid="ignore")
    new_sig[(new_sig < self._low) | (new_sig > self._high)] = np.nan
    np.seterr(invalid=err_set["invalid"])
    return new_sig

def lcm(*numbers):
    """Return lowest common multiple."""
    def lcm(a, b):
        return (a * b) // fractions.gcd(a, b)

    return reduce(lcm, numbers, 1)

def anynan(x, axis=None):
    if sparse.issparse(x):
        return np.isnan(x.sum(axis=axis).A.squeeze())
    else:
        return bn.anynan(x, axis)

def allnan(x, axis=None):
    if sparse.issparse(x):
        if axis is None:
            return bn.allnan(x.data)
        else:
            return nancount(x, axis) == x.shape[axis]
    else:
        return bn.allnan(x, axis)

def nancount(x, axis=None):
    if sp_sparse.issparse(x):
        if sp_sparse.isspmatrix_csr(x):
            is_nan = sp_sparse.csr_matrix((np.isnan(x.data), x.indices, x.indptr))
        elif sp_sparse.isspmatrix_csc(x):
            is_nan = sp_sparse.csc_matrix((np.isnan(x.data), x.indices, x.indptr))
        else:
            raise ValueError(
                "Unsupported sparse matrix type for operation 'replace_nan': {}".format(
                    type(x)
                ))

        return is_nan.sum(axis).A.squeeze()
    else:
        return np.isnan(x).sum(axis)

def replace_nan(x, value, start_col, end_col):
    assert(0 <= start_col <= end_col)

    if sp_sparse.issparse(x):
        if sp_sparse.isspmatrix_csr(x):
            should_replace = (start_col <= x.indices) & (x.indices < end_col)
            should_replace &= np.isnan(x.data)
            x.data[should_replace] = value
        elif sp_sparse.isspmatrix_csc(x):
            start = x.indptr[start_col]
            end = None if end_col >= len(x.indptr) else x.indptr[end_col]
            should_replace = np.zeros(len(x.data), np.bool_)
            should_replace[start:end] = True
            should_replace &= np.isnan(x.data)
            x.data[should_replace] = value
        else:
            raise ValueError(
                "Unsupported sparse matrix type for operation 'replace_nan': {}".format(
                    type(x)
                ))
    else:
        bn.replace(x[:, start_col:end_col], np.nan, value)

def replace_infinite(x, value, start_col, end_col):
    assert(0 <= start_col <= end_col)

    if sp_sparse.issparse(x):
        if sp_sparse.isspmatrix_csr(x):
            should_replace = (start_col <= x.indices) & (x.indices < end_col)
            should_replace &= ~np.isfinite(x.data)
            x.data[should_replace] = value
        elif sp_sparse.isspmatrix_csc(x):
            start = x.indptr[start_col]
            end = None if end_col >= len(x.indptr) else x.indptr[end_col]
            should_replace = ~np.isfinite(x.data[start:end])
            x.data[start:end][should_replace] = value
        else:
            raise ValueError(
                "Unsupported sparse matrix type for operation 'replace_nan': {}".format(
                    type(x)
                ))
    else:
        bn.replace(x[:, start_col:end_col], np.PINF, value)
        bn.replace(x[:, start_col:end_col], np.NINF, value)
        bn.replace(x[:, start_col:end_col], np.nan, value)

def pad_between(x, n_values, value):
    """Adds `n_values` copies of `value` after each element in `x`.

    :param x: The original elements, must be 1d
    :param n_values: How many `value` to add after each element
    :param value: The value used for padding
    :return: The padded 1d array
    """
    if x.ndim != 1:
        raise ValueError("Padding is only implemented for 1d arrays but array is {}d".format(x.ndim))

    x_padded = np.full(x.shape[0] * (1 + n_values), value, dtype=x.dtype)
    x_padded[::(1 + n_values)] = x
    return x_padded

def resample(x, up, down, order=6):
    """Resamples the data by the fraction up/down.
    Implemented using zero padding after each sample
    (up - 1) times and then low pass filtering the data.
    Finally taking every down'th sample.
    Implemented similar to the MATLAB decimate.
    Also similar to scipy decimate but using filtfilt instead
    of lfiter for lowpass filtering.

    :param x: The signal to be filtered, must be 1d
    :param up: The interpolation ratio, must be integer
    :param down: The decimation ratio, must be integer
    :param order: The order of the iir filter, default=6
    :return: The resampled 1d array
    """
    up = int(up)
    down = int(down)
    order = int(order)

    if not np.isfinite(x).all():
        raise ValueError("Signal contains infinite values which will corrupt the result when filtering")

    if up > 1:
        x_padded = pad_between(x, up - 1, 0.)
    else:
        x_padded = x

    # Default values taken from MATLAB implementation of decimate
    cutoff = 0.8 / down
    b, a = sp_signal.cheby1(order, 0.05, cutoff)
    x_filtered = sp_signal.filtfilt(b, a, x_padded)
    x_filtered *= up
    return x_filtered[::down]

@numba.jit(nogil=True)
def first_argrelmin(x, allow_zero=False):
    """ Returns the index of the first relative minimum.

    :param x: The 1d array to search in
    :param allow_zero: Set to True to allow index `0` to be a valid return value.
    :return: 1d array with all found indices
    """
    assert(x.ndim == 1)
    n = x.shape[0]
    diff = x[1] - x[0]
    pos_prev = diff > 0.
    neg_prev = diff < 0.
    saddle_prev = not (pos_prev or neg_prev)
    saddle_start = 0

    if allow_zero and pos_prev:
        return 0

    for i in range(1, n - 1):
        diff = x[i + 1] - x[i]
        pos = diff > 0.
        neg = diff < 0.

        if pos and neg_prev:
            if saddle_prev:
                return saddle_start
            else:
                return i

        if pos or neg:
            pos_prev = pos
            neg_prev = neg
            saddle_prev = False
        elif not saddle_prev:
            saddle_prev = True
            saddle_start = i
        else:
            saddle_prev = True

    if allow_zero and not saddle_prev:
        return 0
    else:
        return None


@numba.jit(nogil=True)
def first_argrelmax(x, allow_zero=False):
    """ Returns the index of the first relative maximum.

    :param x: The 1d array to search in
    :param allow_zero: Set to True to allow index `0` to be a valid return value.
    :return: 1d array with all found indices
    """
    assert(x.ndim == 1)
    n = x.shape[0]
    diff = x[1] - x[0]
    pos_prev = diff > 0.
    neg_prev = diff < 0.
    saddle_prev = not (pos_prev or neg_prev)
    saddle_start = 0

    if allow_zero and neg_prev:
        return 0

    for i in range(1, n - 1):
        diff = x[i + 1] - x[i]
        pos = diff > 0.
        neg = diff < 0.

        if neg and pos_prev:
            if saddle_prev:
                return saddle_start
            else:
                return i

        if pos or neg:
            pos_prev = pos
            neg_prev = neg
            saddle_prev = False
        elif not saddle_prev:
            saddle_prev = True
            saddle_start = i
        else:
            saddle_prev = True

    if allow_zero and not saddle_prev:
        return 0
    else:
        return None

@numba.jit(nogil=True)
def _min_i(a, b):
    return a if a <= b else b

@numba.jit(nogil=True)
def _min_f(a, b):
    if np.isfinite(a) and np.isfinite(b):
        return a if a <= b else b
    else:
        return np.nan

@numba.jit(nogil=True)
def _max_i(a, b):
    return a if a >= b else b

@numba.jit(nogil=True)
def _max_f(a, b):
    if np.isfinite(a) and np.isfinite(b):
        return a if a >= b else b
    else:
        return np.nan

@numba.jit(nogil=True)
def _correlation_a1(n, x_sum, y_sum, xy_sum, x2_sum):
    num = ((n * xy_sum) - (x_sum * y_sum))
    denom = ((n * x2_sum) - (x_sum * x_sum))
    return num / denom if denom != 0. else np.nan

@numba.jit(nogil=True)
def _correlation_a0(n, x_sum, y_sum, a1):
    num = (y_sum - (a1 * x_sum))
    return num / n if n != 0 else np.nan

@numba.jit(nogil=True)
def _correlation_r(n, x_sum, y_sum, xy_sum, x2_sum, y2_sum):
    num = ((n * xy_sum) - (x_sum * y_sum))
    denom1 = np.sqrt((n * x2_sum) - (x_sum * x_sum))
    denom2 = np.sqrt((n * y2_sum) - (y_sum * y_sum))
    denom = (denom1 * denom2)
    return num / denom if denom != 0. else np.nan

@numba.jit(nogil=True)
def expand_regression(y, left_bound, right_bound, min_correlation=.999):
    """Expands the regression line through `left_bound` and `right_bound` until
    the correlation is less than `min_correlation`. The bounds are synchronously
    incremented by 1 in every step.

    :param y: The 1d array containing the values
    :param left_bound: The starting left bound of the regression line
    :param right_bound: The starting right bound of the regression line
    :param min_correlation: The correlation coefficient the regression
                            line needs to have.
    :return: (correlation_found, correlation, slope, intercept, left_bound, right_bound)
    """
    win_len = y.shape[0]
    slope = np.nan
    icept = np.nan
    left_bound = _min_i(_max_i(left_bound, 0), win_len - 1)
    right_bound = _min_i(_max_i(right_bound, 0), win_len - 1)

    n = float(right_bound - left_bound)
    x_sum = 0.
    y_sum = 0.
    x2_sum = 0.
    y2_sum = 0.
    xy_sum = 0.

    for i in range(left_bound, right_bound+1):
        x_sum += i
        x2_sum += i * i
        y_sum += y[i]
        y2_sum += y[i] * y[i]
        xy_sum += i * y[i]

    r = _correlation_r(n, x_sum, y_sum, xy_sum, x2_sum, y2_sum)

    if r < min_correlation:
        return False, slope, icept, r, left_bound, right_bound

    slope = _correlation_a1(n, x_sum, y_sum, xy_sum, x2_sum)
    icept = _correlation_a0(n, x_sum, y_sum, slope)

    # WHILE: Grow the window centered on the point of max. derivative
    while left_bound >= 0 and right_bound < win_len:
        left_bound -= 1
        right_bound += 1

        n += 2
        x_sum += left_bound
        x_sum += right_bound
        x2_sum += left_bound * left_bound
        x2_sum += right_bound * right_bound
        y_sum += y[left_bound]
        y2_sum += y[left_bound] * y[left_bound]
        y_sum += y[right_bound]
        y2_sum += y[right_bound] * y[right_bound]
        xy_sum += left_bound * y[left_bound]
        xy_sum += right_bound * y[right_bound]

        r_new = _correlation_r(n, x_sum, y_sum, xy_sum, x2_sum, y2_sum)
        slope_new = _correlation_a1(n, x_sum, y_sum, xy_sum, x2_sum)
        icept_new = _correlation_a0(n, x_sum, y_sum, slope)

        # IF: Line does not correlate with rising edge anymore?
        if r < min_correlation or np.isnan(r_new) or np.isnan(slope_new) or np.isnan(icept_new):
            left_bound += 1
            right_bound -= 1
            break
        else:
            r = r_new
            slope = slope_new
            icept = icept_new
        # ENDIF

    return True, slope, icept, r, left_bound, right_bound

def linregress(x, y=None):
    """
    Calculate a regression line

    This computes a least-squares regression for two sets of measurements.

    !!COPY!! of scipy.stats.linregress that is faster for the measured setting
    of small vectors.

    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.

    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    rvalue : float
        correlation coefficient

    """
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = ("If only `x` is given as input, it has to be of shape "
                   "(2, N) or (N, 2), provided shape was %s" % str(x.shape))
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    xmean = np.mean(x, None)
    ymean = np.mean(y, None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

    slope = r_num / ssxm
    intercept = ymean - slope * xmean
    return slope, intercept, r

def unrollaxis(x, i, j=0):
    if i > j:
        return np.rollaxis(x, j, i + 1)
    elif i == 0 and j == 0:
        return x
    else:
        return np.rollaxis(x, j - 1, i)

def apply_along_axis_n(func1d, axis, arrs, *args, **kwargs):
    arrs = list(map(np.asarray, arrs))
    arr = arrs[0]
    nd = arr.ndim
    shape = arr.shape
    has_args = len(args) > 0
    has_kwargs = len(kwargs) > 0
    has_args_and_kwargs = has_args and has_kwargs
    assert(all(arr.shape == shape for arr in arrs))

    if axis < 0:
        axis += nd
    if axis >= nd:
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
                         % (axis, nd))
    # `ind` iterates over the source array
    # I guess `ind` is a Python list because that can be converted
    # to a tuple for indexing faster. In addition, all its contents
    # are integers.
    ind = [0]*(nd-1)
    # `i` also iterates over the source array but contains the slice object.
    # The updates to `ind` are transferred to `i` before reading
    # the source array.
    i = np.zeros(nd, 'O')
    i[axis] = slice(None, None)
    # `indlist` is the list of dimensions that we iterate over in the source array
    indlist = list(range(nd))
    indlist.remove(axis)
    indlist = np.array(indlist)
    # `outshape` is identical to the source array shape except that `axis` is removed
    outshape = np.asarray(arr.shape).take(indlist)
    # set all other source array indices to 0
    i.put(indlist, ind)
    i_tuple = tuple(i.tolist())
    # compute the first result
    res = func1d(*itertools.chain((arr[i_tuple] for arr in arrs), args), **kwargs)

    #  if res is a number, then we have a smaller output array
    if np.isscalar(res):
        # create space for results
        outarr = np.empty(outshape, np.asarray(res).dtype)
        # and set the first result
        outarr[tuple(ind)] = res
        # compute the total number of iterations over all dimensions
        Ntot = np.product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            # increment the next higher dimension if the current dimension
            # is at the end
            while (ind[n] >= outshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            # write to the indexing object
            i.put(indlist, ind)
            i_tuple = tuple(i.tolist())
            wins = (arr[i_tuple] for arr in arrs)

            if has_args_and_kwargs:
                res = func1d(*itertools.chain(wins, args), **kwargs)
            elif has_kwargs:
                res = func1d(*wins, **kwargs)
            elif has_args:
                res = func1d(*itertools.chain(wins, args))
            else:
                res = func1d(*wins)

            outarr[tuple(ind)] = res
            k += 1
        return outarr
    else:
        # compute the total number of iterations over all dimensions
        Ntot = np.product(outshape)
        holdshape = outshape
        # recompute the outshape
        outshape = list(arr.shape)
        outshape[axis] = len(res)
        outarr = np.empty(outshape, np.asarray(res).dtype)
        # write the first result
        outarr[tuple(i)] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            # increment the next higher dimension if the current dimension
            # is at the end
            while (ind[n] >= holdshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            # write to the indexing object
            i.put(indlist, ind)
            i_tuple = tuple(i.tolist())
            wins = (arr[i_tuple] for arr in arrs)

            if has_args_and_kwargs:
                res = func1d(*itertools.chain(wins, args), **kwargs)
            elif has_kwargs:
                res = func1d(*wins, **kwargs)
            elif has_args:
                res = func1d(*itertools.chain(wins, args))
            else:
                res = func1d(*wins)

            outarr[i_tuple] = res
            k += 1
        return outarr

def num_window_segments(n, nperseg, noverlap):
    nperseg = int(nperseg)
    noverlap = int(noverlap)

    if noverlap >= nperseg:
        raise ValueError("nperseg must be smaller than n but is not, {} >= {}".format(nperseg, n))

    if noverlap >= nperseg:
        raise ValueError("noverlap must be smaller than nperseg but is not, {} >= {}".format(noverlap, nperseg))

    step = nperseg - noverlap
    return (n - noverlap) // step

def window_reshape(x, nperseg, noverlap=0, axis=0):
    # stackoverflow: Repeat NumPy array without replicating data?
    # <http://stackoverflow.com/a/5568169>
    nperseg = int(nperseg)
    noverlap = int(noverlap)

    if noverlap >= nperseg:
        raise ValueError("noverlap={} must be smaller than nperseg={}".format(noverlap, nperseg))

    if x.shape[axis] < nperseg:
        raise ValueError("nperseg={} is bigger than the number of elements={}".format(nperseg, x.shape[-1]))

    needs_roll = x.ndim > 1

    if needs_roll:
        x = np.rollaxis(x, -1, axis)

    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    if needs_roll:
        result = unrollaxis(result, len(result.shape)-1, axis)

    return result

def repeat_view(x, shape):
    """Broadcast unit dimensions in `x` to match those in `ref` without copies"""
    x = np.asarray(x)
    if x.ndim == 1 and x.shape[0] == 1:
        strides = [0] * len(shape)
    else:
        strides = [
            0 if x.shape[j] == 1 else x.strides[j]
            for j in range(x.ndim)
        ] + [0] * len(shape)
        shape = list(x.shape) + list(shape)
    from numpy.lib.stride_tricks import as_strided
    return as_strided(x, shape=shape, strides=strides)

def broadcast_view(x, ref):
    """Broadcast unit dimensions in `x` to match those in `ref` without copies"""
    assert(x.ndim == ref.ndim)
    strides = [0 if x.shape[j] == 1 else x.strides[j] for j in range(x.ndim)]
    shape = [ref.shape[j] if x.shape[j] == 1 else x.shape[j] for j in range(x.ndim)]
    from numpy.lib.stride_tricks import as_strided
    return as_strided(x, shape=shape, strides=strides)

def align_at(src, dst, start, stop, pulse, loc, n):
    src_start = max(pulse - loc, start)
    src_stop = min(pulse - loc + n, stop)
    src_len = src_stop - src_start
    offset = max(0, start - (pulse - loc))
    dst[:] = 0.

    if src_len <= 0:
        return

    dst_idx = slice(offset, offset+src_len)
    dst[dst_idx] = src[src_start:src_stop]
    dst[dst_idx] -= np.min(dst[dst_idx])

def to_freq(x, nperseg=4096, noverlap=0):
    x_win = window_reshape(x, nperseg, noverlap)
    x_fft = np.fft.fft(x_win)

    if nperseg % 2 == 1:
        x_len = (nperseg + 1) // 2
    else:
        x_len = nperseg // 2 + 1

    x_fft = x_fft[..., :x_len]
    x_fft[..., -1] = np.conjugate(x_fft[..., -1])
    return x_fft

def to_sig(x_fft):
    assert(x_fft.shape[-1] % 2 == 1)
    x_fft = np.concatenate([x_fft[..., :-1], np.conjugate(x_fft[..., :0:-1])], axis=-1)
    x_win = np.fft.ifft(x_fft)
    x = x_win.flatten()
    return x

def circular_delay(a, b, h=None):
    # http://stackoverflow.com/questions/4688715/find-time-shift-between-two-similar-waveforms
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    Ar = -A.conjugate()  # Either -conjugate or compute fft of reverse of a
    if h is None:
        return np.argmax(np.abs(np.fft.ifft(Ar * B)))
    else:
        return np.argmax(np.abs(np.fft.ifft(Ar * B * h)))

def transfer_function_estimate(x, y, fs, welch_nperseg=2048):
    f, Pxy = sp_signal.csd(x, y, nperseg=welch_nperseg, fs=fs, scaling='spectrum')
    _, Pxx = sp_signal.csd(x, x, nperseg=welch_nperseg, fs=fs, scaling='spectrum')
    _, Pyy = sp_signal.csd(y, y, nperseg=welch_nperseg, fs=fs, scaling='spectrum')
    coh = np.divide(np.abs(Pxy), np.sqrt(Pxx * Pyy))
    H = np.divide(Pxy, Pxx)
    mag = np.abs(H)
    phase = np.arctan2(H.real, H.imag)
    return f, coh, mag, phase

def tree_iter_post(node, children):
    """ Yields an iterator to the postorder iteration on the tree rooted at `node`.

    :param node: The root of the tree to iterate
    :param children: A function returning the children of a given node
    :type children: (T) -> list[T]
    """
    for sub_node in itertools.chain(*(tree_iter_post(child, children) for child in children(node))):
        yield sub_node

    yield node

def tree_map_post(node, children, map_fun):
    """

    :param node: The root of the tree to map
    :type node: T
    :param children: A function returning the children of a given node
    :type children: (T) -> list[T]
    :param map_fun: A function receiving the current node and its mapped children
    :type map_fun: (T, list[U]) -> U
    """
    mapped_children = [
        tree_map_post(child, children, map_fun)
        for child in children(node)
    ]

    return map_fun(node, mapped_children)

@numba.jit(nogil=True)
def searchsorted(arr, key):
    i_min = 0
    i_max = arr.shape[0] - 1

    if key < arr[i_min]:
        return i_min
    elif arr[i_max] < key:
        return i_max + 1

    while i_min < i_max:
        # calculate the midpoint for roughly equal partition
        i_mid = (i_min + i_max) // 2

        if arr[i_mid] == key:
            # key found at index i_mid
            return i_mid
        # determine which subarray to search
        elif arr[i_mid] < key:
            # change min index to search upper subarray
            i_min = i_mid + 1
        else:
            # change max index to search lower subarray
            i_max = i_mid - 1

    # key was not found
    # case i_min - i_max == 1
    #   i_mid = i_min
    #   arr[i_mid] < key: i_min = i_max and result = i_max
    #   arr[i_mid] > key: i_max = i_min - 1 and result = i_min
    #   --> result = max(i_min, i_max)
    return max(i_min, i_max)

@numba.jit(nogil=True)
def standardize(arr, copy=True, axis=0, eps=1e-7):
    assert arr.ndim <= 2
    if axis < 0:
        axis += arr.ndim

    assert axis >= 0
    assert axis < arr.ndim

    if copy:
        arr = arr.copy()

    if axis == 0:
        arr = arr.T

    for i in range(arr.shape[0]):
        mean = np.mean(arr[i])

        for j in range(arr.shape[1]):
            arr[i, j] -= mean

        std = np.std(arr[i])

        if abs(std) < eps:
            continue

        for j in range(arr.shape[1]):
            arr[i, j] /= std

    if axis == 0:
        arr = arr.T

    return arr

@numba.jit(nogil=True)
def _quicksort_rec(data, idx, lo, hi):
    if hi - lo > 100:
        p = _partition(data, idx, lo, hi)

        _quicksort_rec(data, idx, lo, p)
        _quicksort_rec(data, idx, p + 1, hi)
    else:
        _insertsort(data, idx, lo, hi)

@numba.jit(nogil=True)
def quicksort(data, idx):
    _quicksort_rec(data, idx, 0, data.shape[0] - 1)

@numba.jit(nogil=True)
def _median_of_three(idx, lo, hi):
    mid = (hi + lo) // 2

    a = idx[lo]
    b = idx[mid]
    c = idx[hi]

    if a > b:
        if b > c:
            return b, mid
        elif a > c:
            return c, hi
        else:
            return a, lo
    else:
        if a > c:
            return a, lo
        elif b > c:
            return c, hi
        else:
            return b, mid

@numba.jit(nogil=True)
def _swap(data, i, j):
    tmp = data[i].copy()
    data[i] = data[j]
    data[j] = tmp

@numba.jit(nogil=True)
def _partition(data, idx, lo, hi):
    # pivot, pivot_ix = _median_of_three(idx, lo, hi)
    #
    # idx[pivot_ix], idx[lo] = idx[lo], idx[pivot_ix]
    # data[pivot_ix], data[lo] = data[lo], data[pivot_ix]
    pivot = idx[lo]

    i = lo
    j = hi

    while True:
        while idx[j] > pivot:
            j -= 1

        while idx[i] < pivot:
            i += 1

        if i < j:
            _swap(idx, i, j)
            _swap(data, i, j)
        else:
            return j

@numba.jit(nogil=True)
def insertsort(data, idx):
    _insertsort(data, idx, 0, data.shape[0] - 1)

@numba.jit(nogil=True)
def _insertsort(data, idx, lo, hi):
    for i in range(lo + 1, hi + 1):
        x = idx[i].copy()
        y = data[i].copy()
        j = i

        while j > lo and idx[j-1] > x:
            _swap(idx, j, j - 1)
            _swap(data, j, j - 1)

            j -= 1

        idx[j] = x
        data[j] = y

@numba.jit(nogil=True)
def _shuffle(data, idx, n):
    # Shuffle an array a of n elements (indices 0..n-1)
    # Fisher-Yates shuffle, modern algorithm from Wikipedia
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i+1)
        _swap(idx, i, j)
        _swap(data, i, j)

@contextlib.contextmanager
def shuffled(data, start=0, stop=None):
    if stop is None:
        stop = data.shape[0]

    assert(0 <= start)
    assert(start <= stop)
    assert(stop <= data.shape[0])
    n = stop - start
    idx = np.arange(n, dtype=int)
    _shuffle(data[start:stop], idx, n)
    yield data
    quicksort(data, idx)

def contiguous_slices(index):
    assert(index.ndim == 1)
    n_rows = index.shape[0]

    if n_rows == 0:
        return

    curr = index[0]
    start = 0

    for i in xrange(n_rows):
        if index[i] == curr:
            continue

        yield slice(start, i, 1)
        curr = index[i]
        start = i

    yield slice(start, None, 1)

def arr_prefilled(sz,fill_val):
    ''' Returns a NP array prefilled with a specified value'''
    new_arr=np.empty(sz)
    new_arr.fill(fill_val)
    return new_arr

def nan_ratio(arr):
    ''' Returns the NAN ratio of an array '''
    nan_count=np.sum(np.isnan(arr))
    return nan_count,nan_count/arr.size
