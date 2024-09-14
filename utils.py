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

def give_idx_time_point(time_array,time_point):
    """
    In a time array, give idx that is closer to the value time_point
    Input:
    time_array: one dimensional array with time points
    time_point: float
    Output:
    idx: int, index that gives time_array[idx] the closest to time_point
    """
    idx = np.abs(time_array - time_point).argmin()
    return idx

def give_idx_time_point_for_different_time_arrays(list_time_arrays,time_point):
    """
    Give idx that is closer to the value time_point in each time array in a list
    Input:
    list_time_arrays: list of one dimensional arrays with time points
    time_point: float
    Output:
    idx_list: list of ints, for each element i gives index that gives list_time_arrays[i][idx] the closest to time_point
    """
    idx_list = []
    for time in list_time_arrays:
        idx_list.append(give_idx_time_point(time_array=time,time_point=time_point))
    return idx_list

def give_idx_ppm_point(ppm_array,ppm_point):
    """
    In a flipped ppm array, give idx that is closer to the value ppm_point
    Input:
    ppm_array: one dimensional array with ppm points
    ppm_point: float
    Details:
    ppm_array organized from (most positive value,most negative value) (convention in MRS)
    However in the spectrogram the lines are counted from the most negative freq to the most positive
    Output:
    idx: int, index that gives np.flip(ppm_array)[idx] the closest to ppm_point, therefore: the corresponding spectrogram, will have
        the line idx as the one that corresponds to the ppm "ppm_point"
    """
    idx = np.abs(np.flip(ppm_array) - ppm_point).argmin()
    return idx

def give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays,ppm_point):
    """
    Give idx that is closer to the value ppm_point in each ppm array in a list
    Input:
    list_ppm_arrays: list of one dimensional arrays with ppm points
    ppm_point: float
    Output:
    idx_list: list of ints, for each element i gives index that gives np.flip(list_ppm_arrays[i])[idx] the closest to ppm_point
    """
    idx_list = []
    for ppm in list_ppm_arrays:
        idx_list.append(give_idx_ppm_point(ppm_array=ppm,ppm_point=ppm_point))
    return idx_list


def detect_minmax_in_proximity_1d(signal,peak_idx,peak_idx_plus,peak_idx_minus,preference):
    """
    Find a local minimum or maximum in an one dimensional array.
    Assumes peak_idx as a first guess for the position of this min/max.
    Assumes 0 as the baseline.
    Inputs:
    signal: one dimensional array, the signal we are looking for a local min/max
    peak_idx: int, signal[peak_idx] is a likely point for the min/max, we are searching around it
    peak_idx_plus: int, maximum idx where to look, peak_idx < peak_idx_plus
    peak_idx_minus: int, minimum idx where to look, peak_idx > peak_idx_minus
    preference: str, indicate if we are looking in preference for a peak ('positive') or for a valley ('negative')
    Outputs: 
    dict_: dictionary with result and with variable to check the function behaviour 
            keys: 'highest': idx for the local min/max
            'left': [we found a possible min/max at left of peak_idx,idx of this prospective value]
            'right': [we found a possible min/max at right of peak_idx,idx of this prospective value]
    """
    keep_going_left = True
    index_left = np.arange(peak_idx_minus,peak_idx+1)
    #idx counter 
    count_left = 0
    #we are in the first idx at left of peak_idx
    first_left = True
    #we found a peak or valley
    found_peak_left = False
    while keep_going_left == True and np.abs(-2-count_left) <= len(index_left):
        #searches from the closest to peak_idx to the furtherest
        deriv = signal[index_left[-2-count_left]] - signal[index_left[-1-count_left]]
        if first_left == True:
            #at the point before peak_idx is the signal rising or is it falling?
            deriv_signal = (deriv > 0)
            first_left = False
            count_left = count_left + 1
        else:
            if ((deriv > 0) == True and (deriv_signal == True)) or ((deriv > 0) == False and (deriv_signal == False)):
                #does the signal keeps its direction? if yes keep advancing
                count_left = count_left + 1
            else:
                #if not, the signal changed direction, and signal[index_left[-1-count_left]] may be a valley or a peak
                if np.abs(-3-count_left) <= len(index_left):
                    deriv_aux = signal[index_left[-3-count_left]] - signal[index_left[-2-count_left]]
                    if (deriv*deriv_aux) > 0:
                        #if this new direction is kept at the point signal[index_left[-2-count_left]], then signal[index_left[-1-count_left]]
                        #is a peak or a valley and we can stop looking
                        peak_left  = index_left[-1-count_left]
                        keep_going_left = False
                        found_peak_left = True
                    else:
                        #if the direction changes, then the direction changing on the previous point was problably just due to noise
                        count_left = count_left + 1
                        deriv_signal = (deriv > 0)
                else:
                    #if there is no more testing... well, then signal[index_left[-1-count_left]] is our valley or peak
                    peak_left  = index_left[-1-count_left]
                    keep_going_left = False
                    found_peak_left = True
    #if we got here and found_peak_left = False, then probably peak_idx is a local maximum or local minimum
    #or the minimum/max is outside the region of interest              

    keep_going_right = True
    index_right = np.arange(peak_idx,peak_idx_plus)
    #idx counter 
    count_right = 0
    #we are in the first idx at right of peak_idx
    first_right = True
    #we found a peak or valley
    found_peak_right = False
    while keep_going_right == True and np.abs(count_right+1) <= len(index_right)-1:
        #searches from the closest to peak_idx to the furtherest
        deriv = signal[index_right[count_right+1]] - signal[index_right[count_right]]
        if first_right == True:
            #at the point after peak_idx is the signal rising or is it falling?
            deriv_signal = (deriv > 0)
            first_right = False
            count_right = count_right + 1
        else:
            if ((deriv > 0) == True and (deriv_signal == True)) or ((deriv > 0) == False and (deriv_signal == False)):
                #does the signal keeps its direction? if yes keep advancing
                count_right = count_right + 1
            else:
                #se mudou de direção...
                if count_right+2 <= len(index_right)-1:
                    deriv_aux = signal[index_right[count_right+2]] - signal[index_right[count_right+1]]
                    if (deriv*deriv_aux) > 0:
                        #if this new direction is kept at the point signal[index_right[count_right+1]], then signal[index_right[count_right+1]]
                        #is a peak or a valley and we can stop looking
                        peak_right  = index_right[count_right]
                        keep_going_right = False
                        found_peak_right = True
                    else:
                        #if the direction changes, then the direction changing on the previous point was problably just due to noise
                        count_right = count_right + 1
                        deriv_signal = (deriv > 0)
                else:
                    #if there is no more testing... well, then signal[index_right[count_right+1]] is our valley or peak
                    peak_right  = index_right[count_right]
                    keep_going_right = False
                    found_peak_right = True
    #if we got here and found_peak_right = False, then probably peak_idx is a local maximum or local minimum
    #or the minimum/max is outside the region of interest    
                    
    #if there were prospective local min/max found on both sides of peak_idx...
    if (found_peak_left == True) and (found_peak_right == True):
        if preference == 'positive':
            #if we are looking for a peak
            if signal[peak_idx] <= 0 and signal[peak_left] <= 0 and signal[peak_right] <= 0:
                #and all possible peaks we found were negative values
                #then the point closest to zero is our peak
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_left],signal[peak_right]])))
            else:
                #but if it isn't the case, then the point with highest value is our peak
                aux = np.argmax(np.array([signal[peak_idx],signal[peak_left],signal[peak_right]]))
        elif preference == 'negative':
            #if we are looking for a valley
            if signal[peak_idx] >= 0 and signal[peak_left] >= 0 and signal[peak_right] >= 0:
                #and all possible valleys are positive values
                #then the point closest to zero is our valley
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_left],signal[peak_right]])))
            else:
                #but if it isn't the case, then the point with the lowest value is the valley
                aux = np.argmin(np.array([signal[peak_idx],signal[peak_left],signal[peak_right]]))
        else:
            #if there is no specified preference, we get the maximum magnitude
            aux = np.argmax(np.abs(np.array([signal[peak_idx],signal[peak_left],signal[peak_right]])))
        #get the correct idx:
        if aux == 0:
            peak_found = peak_idx
        elif aux == 1: 
            peak_found = peak_left
        else:
            peak_found = peak_right
        dict_ = {'highest': peak_found, 'left': [found_peak_left,peak_left], 'right': [found_peak_right,peak_right]} 
    
    #repeated logic, but for the case, where we only find prospective min/max on the left side
    elif (found_peak_left == True) and (found_peak_right == False):
        if preference == 'positive':
            if signal[peak_idx] <= 0 and signal[peak_left] <= 0:
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_left]])))
            else:
                aux = np.argmax(np.array([signal[peak_idx],signal[peak_left]]))
        elif preference == 'negative':
            if signal[peak_idx] >= 0 and signal[peak_left] >= 0:
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_left]])))
            else:
                aux = np.argmin(np.array([signal[peak_idx],signal[peak_left]]))
        else:
            aux = np.argmax(np.abs(np.array([signal[peak_idx],signal[peak_left]])))
        if aux == 0:
            peak_found = peak_idx
        else:
            peak_found = peak_left
        dict_ = {'highest': peak_found, 'left': [found_peak_left,peak_left], 'right': [found_peak_right,'.']} 

    #repeated logic, but for the case, where we only find prospective min/max on the right side  
    elif (found_peak_left == False) and (found_peak_right == True):
        #print(peak_idx,peak_right)
        if preference == 'positive':
            if signal[peak_idx] <= 0 and signal[peak_right] <= 0:
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_right]])))
            else:
                aux = np.argmax(np.array([signal[peak_idx],signal[peak_right]]))
        elif preference == 'negative':
            if signal[peak_idx] >= 0 and signal[peak_right] >= 0:
                aux = np.argmin(np.abs(np.array([signal[peak_idx],signal[peak_right]])))
            else:
                aux = np.argmin(np.array([signal[peak_idx],signal[peak_right]]))
        else:
            aux = np.argmax(np.abs(np.array([signal[peak_idx],signal[peak_right]])))
        
        if aux == 0:
            peak_found = peak_idx
        else:
            peak_found = peak_right
        dict_ = {'highest': peak_found, 'left': [found_peak_left,'.'], 'right': [found_peak_right,peak_right]}
    
    #we found nothing, so in the region of interest, the most likely peak or valley is peak_idx
    else:
        peak_found = peak_idx
        dict_ = {'highest': peak_found, 'left': [found_peak_left,'.'], 'right': [found_peak_right,'.']} 
    
    return dict_

def get_fwhm_1d(signal,peak_idx,peak_idx_plus,peak_idx_minus,preference):
    """
    Get full-width-at-half-maximum of a one dimensional signal for a peak or valley that is assumed close to peak_idx
    Assumes 0 as the baseline.
    Inputs:
    signal: one dimensional array, the signal we are looking for a local min/max
    peak_idx: int, signal[peak_idx] is a likely point for the min/max, we are searching around it
    peak_idx_plus: int, maximum idx where to look, peak_idx < peak_idx_plus
    peak_idx_minus: int, minimum idx where to look, peak_idx > peak_idx_minus
    preference: str, indicate if we are looking in preference for a peak ('positive') or for a valley ('negative')
    Outputs:
    peak_near_dict: dictionary, the output of detect_minmax_in_proximity_1d for checking purposes
    idx_half_left: int, idx of the half_max at left of the peak/valley
    idx_half_right: int, idx of the half_max at right of the peak/valley
    """
    ####THIS FUNCTION NEEDS SOME ADJUSTMENTS TO BECOME MORE ROBUST IF WE ARE DEALING WITH SIGNALS THAT
    # ARE NOT WELL BEHAVED!!!!!!
    #this isn't the case so it is not a major consern...

    #check in the proximity of peak_idx for the peak or the valley we are looking for
    peak_near_dict = detect_minmax_in_proximity_1d(signal=signal,peak_idx=peak_idx,peak_idx_plus=peak_idx_plus,peak_idx_minus=peak_idx_minus,preference=preference)
    half_max = signal[peak_near_dict['highest']]/2
    
    min_dist_left = 1e8
    idx_half_left = peak_near_dict['highest']-1
    counter_left = 0 
    for i in reversed(range(peak_idx_minus,peak_near_dict['highest'])):
        #from the first point at left of the peak/valley
        if np.abs(signal[i] - half_max) <= min_dist_left:
            #check if the value of the signal at the current point is closer to the half_max
            #then the point we had saved first
            #if yes, we are getting close to the half_max
            min_dist_left = np.abs(signal[i] - half_max)
            idx_half_left = i
            counter_left = 0
        else:
            if np.abs(signal[i]) < np.abs(signal[idx_half_left]):
                #if not and this current point is closer to zero (the baseline) then there is good chance that
                #signal[idx_half_left] is our half_max, we advance counter_left
                #this allows for some oscilations around the half_max value
                counter_left = counter_left + 1
            else:
                #if not, we are changing directions, stop because it may be other local peaks around due to noise
                #half_left is assumed to be idx_half_left
                break
        if counter_left == 2:
            #found half_left
            break
    
    #same logic but for the right side of the peak/valley
    min_dist_right = 1e8
    idx_half_right = peak_near_dict['highest']-1
    counter_right = 0 
    for i in range(peak_near_dict['highest']+1,peak_idx_plus+1):
        if np.abs(signal[i] - half_max) <= min_dist_right:
            min_dist_right = np.abs(signal[i] - half_max)
            idx_half_right = i
            counter_right = 0
        else:
            if np.abs(signal[i]) < np.abs(signal[idx_half_right]):
                counter_right = counter_right + 1
            else:
                break
        if counter_right == 2:
            break

    return peak_near_dict, idx_half_left,idx_half_right

def get_fwhm_in_ppm(signals,peak_idx,peak_idx_plus,peak_idx_minus,ppm,preference):
    """
    Get full-width-at-half-maximum of some signals for a peak or valley that is assumed close to peak_idx
    Assumes 0 as the baseline.
    Measures fwhm in ppm.
    Inputs:
    signals: np array (N, K) (N - number of signals of size K)
    peak_idx: int, signals[i,peak_idx] is a likely point for the min/max for each i, we are searching around it
    peak_idx_plus: int, maximum idx where to look, peak_idx < peak_idx_plus
    peak_idx_minus: int, minimum idx where to look, peak_idx > peak_idx_minus
    preference: str, indicate if we are looking in preference for a peak ('positive') or for a valley ('negative')
    Outputs:
    fwhm: np array (N,5) -> fwhm[ ,0]: the fwhm in ppm, fwhm[ ,1]: the fwhm in index units, fwhm[ ,2]: idx of the half_max at left of the peak/valley
                            fwhm[ ,3]: idx of the half_max at right of the peak/valley, fwhm[ ,4]: idx of the peak/valley
    """
    fwhm = np.zeros((signals.shape[0],5))
    for j in range(signals.shape[0]):
        peak_near_dict, idx_half_left, idx_half_right = get_fwhm_1d(signal=signals[j,:],peak_idx=peak_idx,peak_idx_plus=peak_idx_plus,peak_idx_minus=peak_idx_minus,preference=preference)
        fwhm[j,:] = np.array([np.flip(ppm)[idx_half_right]-np.flip(ppm)[idx_half_left],idx_half_right-idx_half_left,idx_half_left,idx_half_right,peak_near_dict['highest']]) 
    return fwhm

def get_fwhm_in_ppm_for_different_signals(list_signals,list_peak_idx,list_ppm,peak_ppm_plus,peak_ppm_minus,preference):
    """
    Get full-width-at-half-maximum of different multi-dimensional signals for a peak or valley that is assumed close to peak_idx
    Assumes 0 as the baseline.
    Inputs:
    list_signals: list of np arrays of shape (N,k) (k may vary) -> one signal = N signals of size k
    list_peak_idx: list of ints, for each signal where to look for the peak/valley
    list_ppm: list of ppm arrays
    peak_plus: float, the maximum ppm to look for the valley/peak
    peak_minus: float, the minimum ppm to look for the valley/peak
    preference: str, indicate if we are looking in preference for a peak ('positive') or for a valley ('negative')
    Outputs:
    fwhm_list: dict with 'mean' and 'std' keys; list of mean (std) fwhm obtained for each multidimensional signal in list, measured in ppm;
    important_idx_fwhm_list: list of arrays (N,3), for each list element i and each signal j of size k in list_signals[i],
    presents:
        important_idx_fwhm_list[i][j,0]: the idx of half_max at left of peak/valley of signal: list_signals[i][j,:]
        important_idx_fwhm_list[i][j,1]: the idx of half_max at right of peak/valley of signal: list_signals[i][j,:]
        important_idx_fwhm_list[i][j,2]: the idx of peak/valley of signal: list_signals[i][j,:]
    """
    fwhm_list = {'mean':[],'std':[]}
    important_idx_fwhm_list = []
    list_peak_idx_plus = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm,ppm_point=peak_ppm_plus)
    list_peak_idx_minus = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm,ppm_point=peak_ppm_minus)
    for i,signal in enumerate(list_signals):
        aux_fwhm = get_fwhm_in_ppm(signals=signal,peak_idx=list_peak_idx[i],peak_idx_plus=list_peak_idx_plus[i],peak_idx_minus=list_peak_idx_minus[i],ppm=list_ppm[i],preference=preference)
        fwhm_list['mean'].append(np.mean(aux_fwhm[:,0]))
        fwhm_list['std'].append(np.std(aux_fwhm[:,0]))
        important_idx_fwhm_list.append(aux_fwhm[:,2:])
    return fwhm_list, important_idx_fwhm_list


def concatenate_generic(selected_keys,spgram_dict,list_time_idx,fid_idx_plot):
    """
    Concatenate spectrograms obtained with different conditions. Assume all spgrams have the same amount of LINES (column nmbr may vary).
    Inputs:
    selected_keys: list of strs, list with keys of spgram_dict to consider for concatenation
    spgram_dict: dict of lists, dict with keys that include those in selected_keys.
                For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    list_time_idx: list of ints, list_time_idx[0] -> the first concatemated element will be spgram_dict[selected_keys[0]][0][fidx_idx_plot,:,list_time_idx[0]]
                    and so on...
        fid_idx_plot: int, among all possible index from 0 to N-1, which one to consider for the concat
    Output:
    spgram_concat: np array (f, list_time_idx[0] + ... + list_time_idx[-1]), 
                    array containing the concatenated images/spectrograms
    """
    size = 0
    for time_idx in list_time_idx:
        size = size+time_idx
    spgram_concat = np.zeros((spgram_dict[selected_keys[0]][0].shape[1],size)).astype(spgram_dict[selected_keys[0]][0].dtype)
    count = 0
    for i,time_idx in enumerate(list_time_idx):
        spgram_concat[:,count:count+time_idx]  = spgram_dict[selected_keys[i]][0][fid_idx_plot,:,:time_idx]
        count = count+time_idx
    return spgram_concat





