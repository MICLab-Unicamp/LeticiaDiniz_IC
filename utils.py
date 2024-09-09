import numpy as np
from scipy.signal import ShortTimeFFT
from data_corruption import TransientMaker

def create_corrupted_fids(gt,t,std_base,std_var,ntransients):
    """
    Create ntransients from the gt FIDs, adding amplitude noise
    Inputs:
    gt: gt FIDs; np array (N,T,2) (N - number of different transients, T - number of samples in each FID)
    t: time points; np array (N, T)
    std_base: float;
    std_var: float;
    ntransients: number of transients derived from each gt; int
    Details about the noise:
    N values are sampled from U[std_base-std_var,std_base+std_var]: sigma_n
    for each gt_n creates ntransients signals with size T by sampling Normal(0,sigma_n)
    Outputs:
    corrupted_fids: np array (N,T,2,ntransients)
    """
    tm = TransientMaker(fids=gt,t=t,transients=160)
    tm.add_random_amplitude_noise(noise_level_base=std_base,noise_level_scan_var=std_var)
    corrupted_fids = tm.fids
    return corrupted_fids

def normalize_complex_vector_between_minus_one_and_one(complex_array):
    """
    Normalize between [-1,1] the real and imaginary parts of a complex array
    Input:
    complex_array: one dimensional np array
    Output:
    normalized_complex_array: one dimensional np array
    """
    real_parts = complex_array.real
    imaginary_parts = complex_array.imag

    min_real = np.min(real_parts)
    max_real = np.max(real_parts)
    min_imaginary = np.min(imaginary_parts)
    max_imaginary = np.max(imaginary_parts)

    range_real = max_real - min_real
    range_imaginary = max_imaginary - min_imaginary

    normalized_real = (((real_parts - min_real)/range_real)*2)-1
    normalized_imaginary = (((imaginary_parts - min_imaginary)/range_imaginary)*2)-1

    normalized_complex_array = normalized_real + 1j*normalized_imaginary
    return normalized_complex_array

def normalize_complex_vector_min_max(complex_array):
    """
    Normalize between [0,1] the real and imaginary parts of a complex array
    Input:
    complex_array: one dimensional np array
    Output:
    normalized_complex_array: one dimensional np array
    """
    real_parts = complex_array.real
    imaginary_parts = complex_array.imag

    min_real = np.min(real_parts)
    max_real = np.max(real_parts)
    min_imaginary = np.min(imaginary_parts)
    max_imaginary = np.max(imaginary_parts)

    range_real = max_real - min_real
    range_imaginary = max_imaginary - min_imaginary

    normalized_real = ((real_parts - min_real)/range_real)
    normalized_imaginary = ((imaginary_parts - min_imaginary)/range_imaginary)

    normalized_complex_array = normalized_real + 1j*normalized_imaginary
    return normalized_complex_array

def normalize_complex_vector_zscore(complex_array):
    """
    Zscore normalization of the real and imaginary parts of a complex array 
    Input:
    complex_array: one dimensional np array
    Output:
    normalized_complex_array: one dimensional np array
    """
    real_parts = complex_array.real
    imaginary_parts = complex_array.imag

    mean_real = np.mean(real_parts)
    std_real = np.std(real_parts)
    mean_imaginary = np.mean(imaginary_parts)
    std_imaginary = np.std(imaginary_parts)

    normalized_real = (real_parts - mean_real)/std_real
    normalized_imaginary = (imaginary_parts - mean_imaginary)/std_imaginary

    normalized_complex_array = normalized_real + 1j*normalized_imaginary
    return normalized_complex_array

def normalize_complex_vector_abs(complex_array):
    """
    Normalize between (-1,1) the real and imaginary parts of a complex array considering the max magnitude
    Input:
    complex_array: one dimensional np array
    Output:
    normalized_complex_array: one dimensional np array
    """
    normalized_complex_array = complex_array/np.max(np.abs(complex_array))
    return normalized_complex_array

def get_normalized_spectrogram(fids,bandwidth,window,mfft,hop,norm,correct_time,a,b):
    """
    Get normalized spectrogram of fids
    Inputs:
    fids: np array (N, T) (N - number of different transients, T - number of samples in each FID)
    bandwidth: sampling frequency used for the fids (param of ShortTimeFFT)
    window: np array (W); window for the STFT (param of ShortTimeFFT)
    mfft: int; the amount of frequencies in the STFT (param of ShortTimeFFT)
    hop: int; the window step (param of ShortTimeFFT)
    norm: str; if 'm1p1' - norm between [-1,1], if 'zscore' - norm zscore, if 'minmax' - norm beteen (-1,1), else norm abs
    correct_time: if True, time is limited between 0 and 1s
    a: float, slope of the transformation between freq and ppm
    b: float, linear coef. of the transformation between freq and ppm
    Outputs:
    spgram: np array (N, mfft, nt) (nt - number of time windows)
    freq_spect: np array (mfft,), frequencies in spgram
    ppm_spect: np array (mfft,), frequencies in spgram converted to ppm
    t_spect: np array (nt,), time smaples in spgram
    """
    qntty = fids.shape[0]
    SFT = ShortTimeFFT(win=window, hop=hop, fs=bandwidth, mfft=mfft, scale_to='magnitude', fft_mode = 'centered')
    t_lo, t_hi, f_lo, f_hi = SFT.extent(fids.shape[1])
    #array with spectrograms
    spgram = []
    for i in range(qntty):
        aux = SFT.stft(fids[i,:])
        if norm == 'm1p1':
            spgram.append(normalize_complex_vector_between_minus_one_and_one(aux))
        elif norm == 'zscore':
            spgram.append(normalize_complex_vector_zscore(aux))
        elif norm == 'minmax':
            spgram.append(normalize_complex_vector_min_max(aux))
        else:
            spgram.append(normalize_complex_vector_abs(aux))
    spgram = np.array(spgram)
    #frequency array
    freq_spect = np.flip(np.linspace(f_lo,f_hi,mfft))
    #ppm array
    ppm_spect = a*freq_spect+b
    #time array
    t_spect = np.linspace(t_lo,t_hi,spgram.shape[2])

    if correct_time == True:
        zero_idx = np.abs(t_spect - 0.0).argmin()
        one_idx = np.abs(t_spect - 1.0).argmin()
        t_spect = t_spect[zero_idx:one_idx]
        spgram = spgram[:,:,zero_idx:one_idx]
    
    return spgram, freq_spect, ppm_spect, t_spect

def center_bins(bins):
    """
    Get the center values of the intervals in the two-dimensional array 'bins'
    Inputs:
    bins: two-dimensional array (N,K), every line [i,:] contains a set of growing values, and we want to retrieve the center of each interval
    Outputs:
    mean_bins: array of size (N,K-1) containing the center points
    """
    mean_bins = []
    for i in range(bins.shape[0]):
        mean_bins.append([])
        for j in range(bins.shape[1]-1):
            aux = (bins[i,j+1]+bins[i,j])/2
            mean_bins[i].append(aux)
    mean_bins = np.array(mean_bins)
    return mean_bins

def get_histogram(spgram,part,nbins,flatten,normalized):
    """
    Get the histogram of a two or three dimensional array.
    Inputs:
    spgram: an array of size (N,K) or (N,K,W), we take the histogram of [i,:] or [i,:,:]
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    nbins: int, the number of bins to consider for the histogram
    flatten: bool, if True, ignores the differences between the arrays in the dimension 0 (gets the histogram
             considering all the value in the N arrays of size K or K*W)
    normalized: bool, if True, histograms are normalized by the total number of samples
    Outputs:
    hist: array of size (N, bins), containing the histograms of the N signals in spgram
    bins_hist: array of size (N,bins+1), containing the extremes of each bin interval
    bins_centered: array of size (N, bins), containing the certer of every interval in bins
    """

    if part == 'abs':
       obj = np.abs(spgram)
    elif part == 'imag':
       obj = np.imag(spgram)
    elif part == 'phase':
       obj = np.angle(spgram, False)
    else:
       obj = np.real(spgram)
    
    if flatten == True:
        aux, bins_hist = np.histogram(obj.flatten(), nbins)
        if normalized == True:
            hist = aux/aux.sum()  
        else:
            hist = aux
    else:
        hist = []
        bins_hist = []
        for i in range(obj.shape[0]):
            #switched from 200 to 8000, from density to absolute
            if len(obj.shape) > 2:
                aux, bins = np.histogram(obj[i,:,:].flatten(), nbins)
            else:
                aux, bins = np.histogram(obj[i,:], nbins)
            #added this normalization
            if normalized == True:
                aux = aux/aux.sum()            
            hist.append(aux)
            bins_hist.append(bins)
    hist = np.array(hist)
    bins_hist = np.array(bins_hist)
    bins_centered = center_bins(bins_hist)
    return hist, bins_hist, bins_centered

def dict_with_stats(seq_stats,names):
    """
    Get the stats of a sequence of different measures in a comprehensive way.
    Input:
    seq_stats: list or tuple, with arrays with different measures
    names: the names of the measures
    Output:
    metrics: dict with keys = names, for every metrics[name] there is another dict with keys 'mean' and 'std'
            for the mean and std of the metric
    """

    metrics = {}
    for i,value in enumerate(seq_stats):
        metrics[names[i]] = {}
        metrics[names[i]]['mean'] = np.mean(value)
        metrics[names[i]]['std'] = np.std(value)

    return metrics

def get_metrics(list_of_interest):
    """
    Get some stats of a list with arrays.
    Input: 
    list_of_interest: list with array of (probably) different sizes.
    Output:
    dict_metrics: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                  for every key there is another dict with keys 'mean' and 'std' for the mean and std of the metric being considered
    """
    from scipy import stats
    mean_aux = []
    median_aux = []
    std_aux = []
    skew_aux = []
    kurt_aux = []
    
    for i in range(len(list_of_interest)):
        mean_aux.append(np.mean(list_of_interest[i]))
        median_aux.append(np.median(list_of_interest[i]))
        std_aux.append(np.std(list_of_interest[i]))
        skew_aux.append(stats.skew(list_of_interest[i]))
        kurt_aux.append(stats.kurtosis(list_of_interest[i]))
    
    dict_metrics = dict_with_stats(seq_stats=(np.array(mean_aux),np.array(median_aux),np.array(std_aux),
                                              np.array(skew_aux), np.array(kurt_aux)),names=('mean','median',
                                              'std','skewness','kurtosis'))
    return dict_metrics




