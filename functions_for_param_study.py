import numpy as np
import librosa
from utils import get_histogram, dict_with_stats,get_metrics

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
    idx: int, index that gives np.flip(ppm_array)[idx] the closest to ppm_point
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

def concatenate_different_mfft(list_mfft_idx,spgram_dict,time_idx,fid_idx_plot,centered_ppm_last=None,list_idx_of_centered=None):
    """
    Concatenate spectrograms obtained with different mffts, therefore, with different qntty of lines.
    Inputs:
    list_mfft_idx: list of ints, list containing the values of mfft's to be considered in the concatenation
    spgram_dict: dict of lists, dict with keys 'mfft_x', where x is the mfft value used to generate the spectrogram saved in this key.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    time_idx: int, all concatenated spgrams will have all the frequencies (lines) and time (columns) until time_idx (spgram[0,:,time_idx])
    fid_idx_plot: int, among all possible index from 0 to N-1, which one to consider for the concat
    centered_ppm_last,list_idx_of_centered: optional arguments
                                            if centered_ppm_last == None: -> align the concatenated images with the center of the last image, which is assumed to be the one with
                                                                            most lines
                                            else: -> align the concatenated images with the center on the line centered_ppm_last of the last image, which is assumed to be the one with
                                                                            most lines
                                                    in this case, we need list_idx_of_centered which is a list with the indexes of the correspondent line (in terms of frequency/ppm) for 
                                                    all the other images (generated with the mfft values in list_mfft_idx)  
    Output:
    spgram_concat: np array (f_max, n*time_idx) where f_max is the nmbr of lines of the last image and n the qntty of objects in list_mfft_idx, array containing the concatenated images/spectrograms
    """
    size = 0
    count = 0
    for idx in list_mfft_idx:
        aux = spgram_dict['mfft_'+str(idx)][0].shape[1]
        count = count + 1
        if aux > size:
            size = aux
    spgram_concat = np.zeros((size,time_idx*count)).astype(spgram_dict['mfft_'+str(list_mfft_idx[0])][0].dtype)
    count = 0
    if centered_ppm_last == None:
        center = int(size/2)
        for idx in list_mfft_idx:
            aux = spgram_dict['mfft_'+str(idx)][0].shape[1]
            spgram_concat[center-int(aux/2):center+int(aux/2),count:count+time_idx]  = spgram_dict['mfft_'+str(idx)][0][fid_idx_plot,:,:time_idx]
            count = count+time_idx
    else:
        for i,idx in enumerate(list_mfft_idx):
            till_aux = list_idx_of_centered[i]
            above_aux = spgram_dict['mfft_'+str(idx)][0].shape[1] - till_aux
            spgram_concat[centered_ppm_last-till_aux:centered_ppm_last+above_aux,count:count+time_idx]  = spgram_dict['mfft_'+str(idx)][0][fid_idx_plot,:,:time_idx]
            count = count+time_idx
    return spgram_concat


def concatenate_different_hop(list_hop_all,list_hop_concat_idx,spgram_dict,list_time_idx,fid_idx_plot):
    """
    Concatenate spectrograms obtained with different hops, therefore, with different qntty of columns.
    Inputs:
    list_hop_all: list of ints, list with all hops considered in the experiment
    list_hop_concat_idx: list of ints, list of index of the desrides hop values in list_hop_all to be considered in the concatenation
    spgram_dict: dict of lists, dict with keys 'hop_x', where x is the hop value used to generate the spectrogram saved in this key.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    list_time_idx: list of ints, all concatenated spgrams i will have all the frequencies (lines) and time (columns) until time_idx (spgram[0,:,list_time_idx[i]])
    fid_idx_plot: int, among all possible index from 0 to N-1, which one to consider for the concat
    Attention: if hop = 3 has index 5 in list_hop_all, in list_hop_concat_idx we should find the value 5, and in 
                list_time_idx the int corresponding to the time point obtained for a spgram generated with hop = 3 
                should also be in index 5.
    Output:
    spgram_concat: np array (f, time_idx[list_hop_concat_idx[0]]+time_idx[list_hop_concat_idx[1]]
                                + time_idx[list_hop_concat_idx[2]] + ... + time_idx[list_hop_concat_idx[-1]]), 
                    array containing the concatenated images/spectrograms
    """
    size = 0
    for idx in list_hop_concat_idx:
        idx_aux = list_time_idx[idx]
        size = size+idx_aux
    spgram_concat = np.zeros((spgram_dict['hop_'+str(list_hop_all[0])][0].shape[1],size)).astype(spgram_dict['hop_'+str(list_hop_all[0])][0].dtype)
    count = 0
    for idx in list_hop_concat_idx:
        idx_aux = list_time_idx[idx]
        spgram_concat[:,count:count+idx_aux]  = spgram_dict['hop_'+str(list_hop_all[idx])][0][fid_idx_plot,:,:idx_aux]
        count = count+idx_aux
    return spgram_concat

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

def segment_relevant_peaks_dict(spgram_dict,idx_list_1ppm,idx_list_4ppm,idx_list_GABA,idx_list_NAA,idx_list_Glx,idx_time_list_0d4,idx_peaks_regions_limits_dict):
    """
    Get dict with segmented spectrograms in the region from 1 to 4ppm.
    We segment the GABA region, NAA e Glx using different theresholds found empirically.
    Inputs:
    spgram_dict: dict of lists, dict with different keys.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    idx_list_1ppm: list of int, list of index of the lines corresponding to 1ppm to each image in spgram_dict
    idx_list_4ppm: list of int, list of index of the lines corresponding to 4ppm to each image in spgram_dict
    idx_list_GABA: list of int, list of index of the lines corresponding to 3.00ppm to each image in spgram_dict
    idx_list_NAA: list of int, list of index of the lines corresponding to 2.02ppm to each image in spgram_dict
    idx_list_Glx: list of int, list of index of the lines corresponding to 3.75ppm to each image in spgram_dict
    idx_time_list_0d4: list of int, list of index of the columns corresponding to 0.4s to each image in spgram_dict
    idx_peaks_regions_limits_dict: dict of dicts containing arrays
                                   idx_peaks_regions_limits_dict is a dict with the same keys as spgram_dict
                                   For each key in idx_peaks_regions_limits_dict there is an inner dict with keys 'NAA','GABA' and 'Glx'
                                   idx_peaks_regions_limits_dict['key']['NAA'] contains an array of size (N,3), where 
                                   element [j,0] - contains for the j-th FID the line_l where we find the FWHM of the NAA considering line_l < line_NAA_peak
                                   element [j,1] - contains for the j-th FID the line_r where we find the FWHM of the NAA considering line_r > line_NAA_peak
                                   element [j,2] - contains for the j-th FID the line_NAA_peak
                                   Analogous for 'GABA' and 'Glx'.
    Output:
    segm_dict: dict of bool arrays, dict with the same keys as spgram_dict. Each element is an array (N,fn,tn) (fn and tn vary with x) of the segmented image in the region betwee 1 to 4ppm.
    """
    segm_dict = {}
    list_of_keys = list(spgram_dict.keys())
    for i in range(len(list_of_keys)):
        segm_dict[list_of_keys[i]] = np.zeros(spgram_dict[list_of_keys[i]][0][:,idx_list_1ppm[i]:idx_list_4ppm[i]+1,:].shape).astype(bool)
        for j in range(spgram_dict[list_of_keys[0]][0].shape[0]):
            line_minus_GABA = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['GABA'][j,0])
            line_plus_GABA = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['GABA'][j,1])+1
            
            #values larger than the mean of the GABA line
            aux = (np.real(spgram_dict[list_of_keys[i]][0][j,line_minus_GABA:line_plus_GABA,:]) > np.mean(np.real(spgram_dict[list_of_keys[i]][0][j,idx_list_GABA[i],:]))).astype(bool)
            segm_dict[list_of_keys[i]][j,line_minus_GABA-idx_list_1ppm[i]:line_plus_GABA-idx_list_1ppm[i],:] = aux

            line_minus_NAA = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['NAA'][j,0])
            line_plus_NAA = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['NAA'][j,1])+1

            #abs values > mean abs value at the NAA line between 0.2 till 0.4s
            aux = (np.abs(np.real(spgram_dict[list_of_keys[i]][0][j,line_minus_NAA:line_plus_NAA,:])) > np.mean(np.abs(np.real(spgram_dict[list_of_keys[i]][0][j,idx_list_NAA[i],int(idx_time_list_0d4[i]/2):idx_time_list_0d4[i]])))).astype(bool)
            segm_dict[list_of_keys[i]][j,line_minus_NAA-idx_list_1ppm[i]:line_plus_NAA-idx_list_1ppm[i],:] =  aux

            line_minus_Glx = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['Glx'][j,0])
            line_plus_Glx = int(idx_peaks_regions_limits_dict[list_of_keys[i]]['Glx'][j,1])+1

            #abs values > std at the Glx line
            aux = (np.abs(np.real(spgram_dict[list_of_keys[i]][0][j,line_minus_Glx:line_plus_Glx,:])) > np.std(np.real(spgram_dict[list_of_keys[i]][0][j,idx_list_Glx[i],:]))).astype(bool)    
            segm_dict[list_of_keys[i]][j,line_minus_Glx-idx_list_1ppm[i]:line_plus_Glx-idx_list_1ppm[i],:] =  aux
    
    return segm_dict

def get_length_relevant_peaks_for_different_spgrams(segm_dict,spgram_dict,idx_list_1ppm,idx_list_4ppm,idx_list_GABA,idx_list_NAA,idx_list_Glx):
    """
    Get length of peaks NAA, GABA and Glx from segmented images.
    Inputs:
    segm_dict: dict of bool arrays, dict with different keys. Each element in a key is an array (N,fn,tn) (fn and tn vary with x) of the segmented image in the region betwee 1 to 4ppm.
    spgram_dict: dict of lists, dict with the same keys as segm_dict.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    idx_list_1ppm: list of int, list of index of the lines corresponding to 1ppm to each image in spgram_dict
    idx_list_4ppm: list of int, list of index of the lines corresponding to 4ppm to each image in spgram_dict
    idx_list_GABA: list of int, list of index of the lines corresponding to 3.00ppm to each image in spgram_dict
    idx_list_NAA: list of int, list of index of the lines corresponding to 2.02ppm to each image in spgram_dict
    idx_list_Glx: list of int, list of index of the lines corresponding to 3.75ppm to each image in spgram_dict
    Output:
    sum_segment: dict with keys 'NAA', 'Glx' and 'GABA', for each key there is a list containing the mean lentgh of the peak for each tri-dimensional array in spgram_dict
                    there is also a measure of the std of the mean
    """
    sum_segment = {'NAA':{'mean':[],'std':[]},'Glx':{'mean':[],'std':[]},'GABA':{'mean':[],'std':[]}}
    list_of_keys = list(segm_dict.keys())
    for i in range(len(list_of_keys)):
        sum_NAA_m = np.mean(np.sum(segm_dict[list_of_keys[i]][:,idx_list_NAA[i]-idx_list_1ppm[i],:],axis=1))
        sum_NAA_s = np.std(np.sum(segm_dict[list_of_keys[i]][:,idx_list_NAA[i]-idx_list_1ppm[i],:],axis=1))
        sum_GABA_m = np.mean(np.sum(segm_dict[list_of_keys[i]][:,idx_list_GABA[i]-idx_list_1ppm[i],:],axis=1))
        sum_GABA_s = np.std(np.sum(segm_dict[list_of_keys[i]][:,idx_list_GABA[i]-idx_list_1ppm[i],:],axis=1))
        sum_Glx_m = np.mean(np.sum(segm_dict[list_of_keys[i]][:,idx_list_Glx[i]-idx_list_1ppm[i],:],axis=1))
        sum_Glx_s = np.std(np.sum(segm_dict[list_of_keys[i]][:,idx_list_Glx[i]-idx_list_1ppm[i],:],axis=1))

        if sum_Glx_m == 0:
            idx_Glx_plus = give_idx_ppm_point(spgram_dict[list_of_keys[i]][2],4.00)
            idx_Glx_minus = give_idx_ppm_point(spgram_dict[list_of_keys[i]][2],3.50)
            sum_Glx_m = np.mean(np.max(np.sum(segm_dict[list_of_keys[i]][:,idx_Glx_minus-idx_list_1ppm[i]:idx_Glx_plus-idx_list_1ppm[i]+1,:],axis=2),axis=1))
            sum_Glx_s = np.mean(np.max(np.sum(segm_dict[list_of_keys[i]][:,idx_Glx_minus-idx_list_1ppm[i]:idx_Glx_plus-idx_list_1ppm[i]+1,:],axis=2),axis=1))

        sum_segment['NAA']['mean'].append(sum_NAA_m)
        sum_segment['NAA']['std'].append(sum_NAA_s)
        sum_segment['GABA']['mean'].append(sum_GABA_m)
        sum_segment['GABA']['std'].append(sum_GABA_s)
        sum_segment['Glx']['mean'].append(sum_Glx_m)
        sum_segment['Glx']['std'].append(sum_Glx_s)
    
    return sum_segment

def get_zcr_for_relevant_peaks_for_different_spgrams(spgram_dict,idx_list_GABA,idx_list_NAA,idx_list_Glx,idx_time_list_0d4):
    """
    Get zero-crossing-rate (zcr) for the peaks of GABA, NAA and Glx for different spectrograms generated with varying parameters.
    Inputs:
    spgram_dict: dict of lists, dict with keys 'param_x'.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    idx_list_GABA: list of int, list of index of the lines corresponding to 3.00ppm to each image generated with different parameters in spgram_dict
    idx_list_NAA: list of int, list of index of the lines corresponding to 2.02ppm to each image generated with different parameters in spgram_dict
    idx_list_Glx: list of int, list of index of the lines corresponding to 3.75ppm to each image generated with different parameters in spgram_dict
    idx_time_list_0d4: list of int, list of index of the columns corresponding to 0.4s to each image generated with different parameters in spgram_dict
    Output:
    zcr_: dict, dict containing keys 'NAA', 'GABA' and 'Glx', for each key there is an one dimensional array with the mean zcr of each peak for each value of the parameter being varied
                                    and another array for the std
    """
    qntty_param = len(list(spgram_dict.keys()))
    qntty_fids = spgram_dict[list(spgram_dict.keys())[0]][0].shape[0]
    zcr_aux = {'NAA': np.zeros((qntty_param,qntty_fids)),'GABA': np.zeros((qntty_param,qntty_fids)),'Glx': np.zeros((qntty_param,qntty_fids))}
    for i in range(qntty_param):
        aux_size = idx_time_list_0d4[i]
        mean_aux_NAA = np.mean(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][:,idx_list_NAA[i],:idx_time_list_0d4[i]]),axis=1)
        mean_aux_GABA = np.mean(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][:,idx_list_GABA[i],:idx_time_list_0d4[i]]),axis=1)
        mean_aux_Glx = np.mean(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][:,idx_list_Glx[i],:idx_time_list_0d4[i]]),axis=1)
        for j in range(qntty_fids):
            zcr_aux['NAA'][i,j] = librosa.feature.zero_crossing_rate(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][j,idx_list_NAA[i],:idx_time_list_0d4[i]])-mean_aux_NAA[j],frame_length=aux_size)[0][0]
            zcr_aux['GABA'][i,j] = librosa.feature.zero_crossing_rate(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][j,idx_list_GABA[i],:idx_time_list_0d4[i]])-mean_aux_GABA[j],frame_length=aux_size)[0][0]
            zcr_aux['Glx'][i,j] = librosa.feature.zero_crossing_rate(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][j,idx_list_Glx[i],:idx_time_list_0d4[i]])-mean_aux_Glx[j],frame_length=aux_size)[0][0]
    zcr_ = {'NAA': {'mean': np.mean(zcr_aux['NAA'],axis=1),
                    'std':  np.std(zcr_aux['NAA'],axis=1)},
            'GABA': {'mean': np.mean(zcr_aux['GABA'],axis=1),
                    'std': np.std(zcr_aux['GABA'],axis=1)},
            'Glx': {'mean': np.mean(zcr_aux['Glx'],axis=1),
                    'std': np.std(zcr_aux['Glx'],axis=1)}}
    return zcr_

def get_max_mean_std_per_sections(image, ppm, part, sections,idx_time_inf=None, idx_time_sup=None):
    """
    Get max, mean and std of absolute values of each region defined by sections (in ppm).
    Inputs:
    image: np array of size (N,f,t), spectrogram of N fids, f frequencies and t times
    ppm: list of floats, list of ppm's in the spectrogram
    part: str, if 'imag' -> consider imaginary part of image, 'abs' -> consider absolute part of image
                'phase' -> consider phase of image, else consider real part
    sections: list of floats, list containing the ppm values that separe each desired region
    idx_time_inf: optional argument, list of ints, if exists, for each section i defined by the ppm, we will only
                 consider the columns up from idx_time_inf[i]
    idx_time_sup: optional argument, list of ints, if exists, for each section i defined by the ppm, we will only
                 consider the columns till idx_time_sup[i]  
    Output:
    max_sec: dict with lists,the mean value for max found on each region (mean among the N fids), considering the absolute values in the region; and the std of the mean
    mean_sec: dict with lists,the mean value for men found on each region (mean among the N fids), considering the absolute values in the region; and the std of the mean
    std_sec: dict with lists,the mean value for std found on each region (mean among the N fids), considering the absolute values in the region; and the std of the mean
    """
    if part == 'imag':
        obj = np.imag(image)
    elif part == 'abs':
        obj = np.abs(image)
    elif part == 'phase':
        obj = np.angle(image,False)
    else:
        obj = np.real(image)
        
    max_sec = {'mean':[],'std':[]}
    mean_sec = {'mean':[],'std':[]}
    std_sec = {'mean':[],'std':[]}
    for i in range(1,len(sections)):
        idx_freq_inf =  give_idx_ppm_point(ppm,sections[i-1])
        idx_freq_sup =  give_idx_ppm_point(ppm,sections[i])
        if idx_time_inf != None and idx_time_sup != None:
            aux = np.abs(obj[:,idx_freq_inf:idx_freq_sup,idx_time_inf[i-1]:idx_time_sup[i-1]])
        elif idx_time_inf != None and idx_time_sup == None:
            aux = np.abs(obj[:,idx_freq_inf:idx_freq_sup,idx_time_inf[i-1]:])
        elif idx_time_inf == None and idx_time_sup != None:
            aux = np.abs(obj[:,idx_freq_inf:idx_freq_sup,:idx_time_sup[i-1]])
        else:
            aux = np.abs(obj[:,idx_freq_inf:idx_freq_sup,:])

        max_sec['mean'].append(np.mean(np.max(aux,axis=(1,2))))
        max_sec['std'].append(np.std(np.max(aux,axis=(1,2))))
        mean_sec['mean'].append(np.mean(np.mean(aux,axis=(1,2))))
        mean_sec['std'].append(np.std(np.mean(aux,axis=(1,2))))
        std_sec['mean'].append(np.mean(np.std(aux,axis=(1,2))))    
        std_sec['std'].append(np.std(np.std(aux,axis=(1,2))))    

    return max_sec,mean_sec,std_sec

def get_max_mean_std_per_sections_for_different_spgrams(spgram_dict, part, sections,idx_time_list_inf=None, idx_time_list_sup=None):
    """
    Get max, mean and std of absolute values of each region defined by sections (in ppm) in different spectrograms.
    Inputs:
    spgram_dict: dict of lists, dict with keys 'param_x'.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    part: str, if 'imag' -> consider imaginary part of image, 'abs' -> consider absolute part of image
                'phase' -> consider phase of image, else consider real part
    sections: list of floats, list containing the ppm values that separe each desired region
    idx_time_list_inf: optional argument, list of lists with ints, if exists, for each spectrgoram acquired with a different parameter j
                    in each section i defined by the ppm, we will only consider the columns up from idx_time_list_inf[j][i]
    idx_time_list_sup: optional argument, list of lists with ints,if exists, for each spectrgoram acquired with a different parameter j
                    in each section i defined by the ppm, we will only consider the columns till idx_time_list_sup[j][i]
    Output:
    max_sec: dict, keys: if one section goes from x ppm to y ppm, section is named 'x:y'
                    for each key there is a list containing the mean value for max found in that section for each image generated with different parameters
                    and also a list with the std of the mean measure
    mean_sec, std_sec: Analogous for the mean and std
    More details: check description of get_max_mean_std_per_sections
    """
    max_sec = {}
    mean_sec = {}
    std_sec = {}
    for j in range(1,len(sections)):
        max_sec[str(sections[j-1])+':'+str(sections[j])] = {'mean':[],'std':[]}
        mean_sec[str(sections[j-1])+':'+str(sections[j])] = {'mean':[],'std':[]}
        std_sec[str(sections[j-1])+':'+str(sections[j])] = {'mean':[],'std':[]}
    for i in range(len(list(spgram_dict.keys()))):
        if idx_time_list_inf != None and idx_time_list_sup != None:
            max_aux,mean_aux,std_aux = get_max_mean_std_per_sections(image=spgram_dict[list(spgram_dict.keys())[i]][0],ppm=spgram_dict[list(spgram_dict.keys())[i]][2], part=part, sections=sections,idx_time_inf=idx_time_list_inf[i], idx_time_sup=idx_time_list_sup[i])
        elif idx_time_list_inf != None and idx_time_list_sup == None:
            max_aux,mean_aux,std_aux = get_max_mean_std_per_sections(image=spgram_dict[list(spgram_dict.keys())[i]][0],ppm=spgram_dict[list(spgram_dict.keys())[i]][2], part=part, sections=sections,idx_time_inf=idx_time_list_inf[i])
        elif idx_time_list_inf == None and idx_time_list_sup != None:
            max_aux,mean_aux,std_aux = get_max_mean_std_per_sections(image=spgram_dict[list(spgram_dict.keys())[i]][0],ppm=spgram_dict[list(spgram_dict.keys())[i]][2], part=part, sections=sections,idx_time_sup=idx_time_list_sup[i])
        else:
            max_aux,mean_aux,std_aux = get_max_mean_std_per_sections(image=spgram_dict[list(spgram_dict.keys())[i]][0],ppm=spgram_dict[list(spgram_dict.keys())[i]][2], part=part, sections=sections)
        for j in range(1,len(sections)):
            max_sec[str(sections[j-1])+':'+str(sections[j])]['mean'].append(max_aux['mean'][j-1])
            max_sec[str(sections[j-1])+':'+str(sections[j])]['std'].append(max_aux['std'][j-1])
            mean_sec[str(sections[j-1])+':'+str(sections[j])]['mean'].append(mean_aux['mean'][j-1])
            mean_sec[str(sections[j-1])+':'+str(sections[j])]['std'].append(mean_aux['std'][j-1])
            std_sec[str(sections[j-1])+':'+str(sections[j])]['mean'].append(std_aux['mean'][j-1])
            std_sec[str(sections[j-1])+':'+str(sections[j])]['std'].append(std_aux['std'][j-1])
    
    return max_sec,mean_sec,std_sec


def qntty_per_region_histogram(regions, hist, centered_bins):
    """
    Quantification of the amount of pixels with absolute values in some histogram regions.
    Inputs:
    regions: list with POSITIVE floats, let regions = [x, y, z] we will get the qntty of pixels in regions :x, x:y, y:z, z:.
             assumes the values are ordered
    hist: two-dimensional array (N,K), array with histograms
    centered_bins: array of size (N, K), containing the center of every interval in the histogram bins
    Output:
    qntty_per_regions: dict, if  regions = [x, y, z] the dict has keys ':x','x:y','y:z','z:'.
                            qntty_per_regions['x:y'] contains the mean value and std for the qntty of pixels with x < abs value < y for the N histograms of K intervals
    """
    qntty_per_regions = {}
    for k in range(len(regions)+1):
        aux = []
        for j in range(centered_bins.shape[0]):
            if k == 0:
                region_center = np.abs(centered_bins[j,:] - (0)).argmin()
                region_neg = np.abs(centered_bins[j,:] - (-regions[k])).argmin()
                region_pos = np.abs(centered_bins[j,:] - (regions[k])).argmin()
                aux.append(np.sum(np.concatenate((hist[j,region_center:region_pos],hist[j,region_neg:region_center]),axis=0)))
            elif k == len(regions):
                region_neg = np.abs(centered_bins[j,:] - (-regions[k-1])).argmin()
                region_pos = np.abs(centered_bins[j,:] - (regions[k-1])).argmin()
                aux.append(np.sum(np.concatenate((hist[j,region_pos:],hist[j,:region_neg]),axis=0)))
            else:
                region_neg = np.abs(centered_bins[j,:] - (-regions[k])).argmin()
                region_pos = np.abs(centered_bins[j,:] - (regions[k])).argmin()
                region_pr_neg = np.abs(centered_bins[j,:] - (-regions[k-1])).argmin()
                region_pr_pos = np.abs(centered_bins[j,:] - (regions[k-1])).argmin()
                aux.append(np.sum(np.concatenate((hist[j,region_pr_pos:region_pos],hist[j,region_neg:region_pr_neg]),axis=0)))
        if k == 0:
            qntty_per_regions[':'+str(regions[k])] = dict_with_stats(seq_stats=[aux],names=['qntty'])['qntty']
        elif k == len(regions):
            qntty_per_regions[str(regions[k-1])+':'] = dict_with_stats(seq_stats=[aux],names=['qntty'])['qntty']
        else:
            qntty_per_regions[str(regions[k-1])+':'+str(regions[k])] = dict_with_stats(seq_stats=[aux],names=['qntty'])['qntty']

    return qntty_per_regions

def histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict,nbins,part,regions,normalized):
    """
    Get the mean histogram for different images. Get also the qntties of pixels for different regions of the histograms.
    Inputs:
    spgram_dict: dict of lists, dict with keys 'param_x'.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    nbins:  int, the number of bins to consider for the histograms
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    regions: list with POSITIVE floats, let regions = [x, y, z] we will get the qntty of pixels in regions :x, x:y, y:z, z:.
             assumes the values are ordered
    normalized: bool, if True, histograms are normalized by the total number of samples, and instead of qntties of pixels, we get %
    Check description of qntty_per_region_histogram for details.
    Outputs:
    hist: list with mean histograms for the spgrams obtained with different parameters. More explicitly: we get the hist for N arrays of size (f,t) obtained
          with a determined parameter, and then we take the mean of these N arrays. We repeat this for every spgram array in the spgram_dict.
    bins: list, for every N histograms obtained from spgrams of size (f,t), we take the mean of the bins of the histogram
    bins_centered:  list, for every N histograms obtained from spgrams of size (f,t), we take the mean of the center of the histogram bins
    qntty_per_regions: dict, if  regions = [x, y, z] the dict has keys ':x','x:y','y:z','z:'.
                            qntty_per_regions['x:y'] contains a list with the mean value and a list of the std for the qntty of pixels with x < abs value < y for every parameter value
    """
    qntty_per_regions = {}
    hist = []
    bins = []
    bins_centered = []
    for k in range(len(regions)+1):
        if k == 0:
            qntty_per_regions[':'+str(regions[k])] = {'mean':[],'std':[]}
        elif k == len(regions):
            qntty_per_regions[str(regions[k-1])+':'] = {'mean':[],'std':[]}
        else:
            qntty_per_regions[str(regions[k-1])+':'+str(regions[k])] = {'mean':[],'std':[]}

    for i in range(len(list(spgram_dict.keys()))):
        hist_aux, bins_aux, bins_centered_aux = get_histogram(spgram=spgram_dict[list(spgram_dict.keys())[i]][0],part=part,nbins=nbins,flatten=False,normalized=normalized)
        hist.append(np.mean(hist_aux,axis = 0))
        bins.append(np.mean(bins_aux,axis = 0))
        bins_centered.append(np.mean(bins_centered_aux,axis = 0))
        qntty_per_aux_dict = qntty_per_region_histogram(regions=regions, hist=hist_aux, centered_bins=bins_centered_aux)
        for k in range(len(regions)+1):
            if k == 0:
                qntty_per_regions[':'+str(regions[k])]['mean'].append(qntty_per_aux_dict[':'+str(regions[k])]['mean'])
                qntty_per_regions[':'+str(regions[k])]['std'].append(qntty_per_aux_dict[':'+str(regions[k])]['std'])
            elif k == len(regions):
                qntty_per_regions[str(regions[k-1])+':']['mean'].append(qntty_per_aux_dict[str(regions[k-1])+':']['mean'])
                qntty_per_regions[str(regions[k-1])+':']['std'].append(qntty_per_aux_dict[str(regions[k-1])+':']['std'])
            else:
                qntty_per_regions[str(regions[k-1])+':'+str(regions[k])]['mean'].append(qntty_per_aux_dict[str(regions[k-1])+':'+str(regions[k])]['mean'])
                qntty_per_regions[str(regions[k-1])+':'+str(regions[k])]['std'].append(qntty_per_aux_dict[str(regions[k-1])+':'+str(regions[k])]['std'])
    
    return hist, bins, bins_centered, qntty_per_regions


def stats_per_segmented_regions(regions_threshold,spgram, part,not_zero_centered=None,center_value=None):
    """
    Get stats of segmented regions in spgram.
    Inputs:
    regions_threshold: list of floats, list with thresholds (in order) for segmentation
    spgram: array of size (N,f,t)
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    not_zero_centered, center_value (optional): if not_zero_centered is True than subtracts center_value of 
                    the spgram before considering the thresholds (thresholds are assumed to consider a center valued spgram)
    Outputs:
    stats_per_region: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                  for every key there is another dict with keys with the region names
                  if region = [x,y,z] then stats_per_region['mean'] get keys [':x','x:y','y:z','z:']
                  for every region key there is another inner dict with keys 'mean' and 'std' for the mean and std of 
                  the metric being considered
    """
    stats_per_region = {'mean':{},'median':{},'std':{},'skewness':{},'kurtosis':{}}

    if part == 'imag':
        obj = np.imag(spgram)
    elif part == 'abs':
        obj = np.abs(spgram)
    elif part == 'phase':
        obj = np.angle(spgram,False)
    else:
        obj = np.real(spgram)
    
    for k in range(len(regions_threshold)+1):
        aux_list = []
        for j in range(spgram.shape[0]):
            aux = obj[j,:,:]
            if not_zero_centered == True and center_value != None:
                aux_zero_centered = aux - center_value
                if k == 0:
                    seg_ = (np.abs(aux_zero_centered) < regions_threshold[k]).astype('int')
                elif k == len(regions_threshold):
                    seg_ = (np.abs(aux_zero_centered) > regions_threshold[k-1]).astype('int')
                else:
                    seg_ = (np.abs(aux_zero_centered) > regions_threshold[k-1]).astype('int')*(np.abs(aux_zero_centered) < regions_threshold[k]).astype('int')
            else:
                if k == 0:
                    seg_ = (np.abs(aux) < regions_threshold[k]).astype('int')
                elif k == len(regions_threshold):
                    seg_ = (np.abs(aux) > regions_threshold[k-1]).astype('int')
                else:
                    seg_ = (np.abs(aux) > regions_threshold[k-1]).astype('int')*(np.abs(aux) < regions_threshold[k]).astype('int')
            aux_seg = (seg_*aux).ravel()
            if np.all(aux_seg == 0) == False:
                aux_seg = aux_seg[aux_seg != 0]
                aux_list.append(aux_seg)                
            else:
                print('a region without any segmented object has been found, you might want to rethink your threshold if this is not expected')
        dict_metrics_aux = get_metrics(list_of_interest=aux_list)    

        if k == 0:
            name = ':'+str(regions_threshold[k])
        elif k == len(regions_threshold):
            name = str(regions_threshold[k-1])+':'
        else:
            name = str(regions_threshold[k-1])+':'+str(regions_threshold[k])

        stats_per_region['mean'][name] = dict_metrics_aux['mean']
        stats_per_region['median'][name] = dict_metrics_aux['median']
        stats_per_region['std'][name] = dict_metrics_aux['std']
        stats_per_region['skewness'][name] = dict_metrics_aux['skewness']
        stats_per_region['kurtosis'][name] = dict_metrics_aux['kurtosis']

    return stats_per_region

def stats_per_segmented_regions_for_different_spgrams(regions_threshold,spgram_dict, part, not_zero_centered=None,center_value=None):
    """
    Get the stats for the segmented regions in spectrgrams generated by varying parameters.
    Inputs:
    regions_threshold: list of floats, list with thresholds (in order) for segmentation
    spgram_dict: dict of lists, dict with keys 'param_x'.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    not_zero_centered, center_value (optional): if not_zero_centered is True than subtracts center_value of 
                    spgrams before considering the thresholds (thresholds are assumed to consider center valued spgrams)
    Outputs:
    stats_per_region: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                  for every key there is another dict with keys with the region names
                  if region = [x,y,z] then stats_per_region['mean'] get keys [':x','x:y','y:z','z:']
                  for every region key there is a list with the mean and one for the std of 
                  the metric being considered
    """
    stats_per_region = {'mean':{},'median':{},'std':{},'skewness':{},'kurtosis':{}}
    for j in range(len(list(stats_per_region.keys()))):
        stats_per_region[list(stats_per_region.keys())[j]] = {}
        for k in range(len(regions_threshold)+1):
            if k == 0:
                stats_per_region[list(stats_per_region.keys())[j]][':'+str(regions_threshold[k])] = {'mean':[],'std':[]}
            elif k == len(regions_threshold):
                stats_per_region[list(stats_per_region.keys())[j]][str(regions_threshold[k-1])+':'] = {'mean':[],'std':[]}
            else:
                stats_per_region[list(stats_per_region.keys())[j]][str(regions_threshold[k-1])+':'+str(regions_threshold[k])] = {'mean':[],'std':[]}

    for i in range(len(list(spgram_dict.keys()))):
        #print('window',i)
        if not_zero_centered != None and center_value != None:
            stats_per_region_aux = stats_per_segmented_regions(regions_threshold=regions_threshold,spgram=spgram_dict[list(spgram_dict.keys())[i]][0], part=part, not_zero_centered=not_zero_centered,center_value=center_value)
        else:
            stats_per_region_aux = stats_per_segmented_regions(regions_threshold=regions_threshold,spgram=spgram_dict[list(spgram_dict.keys())[i]][0], part=part)

        for k in range(len(regions_threshold)+1):
            if k == 0:
                name = ':'+str(regions_threshold[k])
            elif k == len(regions_threshold):
                name = str(regions_threshold[k-1])+':'
            else:
                name = str(regions_threshold[k-1])+':'+str(regions_threshold[k])

            for j in range(len(list(stats_per_region.keys()))):
                stats_per_region[list(stats_per_region.keys())[j]][name]['mean'].append(stats_per_region_aux[list(stats_per_region.keys())[j]][name]['mean'])
                stats_per_region[list(stats_per_region.keys())[j]][name]['std'].append(stats_per_region_aux[list(stats_per_region.keys())[j]][name]['std'])
    
    return stats_per_region

def stats_global_for_different_spgrams(spgram_dict, part):
    """
    Get the stats for the spectrograms generated by varying parameters.
    Inputs:
    spgram_dict: dict of lists, dict with keys 'param_x'.
                For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    Outputs:
    stats_glb: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                for every key there is a list with the mean and one for the std of 
                the metric being considered
    """
    stats_glb = {'mean':{},'median':{},'std':{},'skewness':{},'kurtosis':{}}
    for j in range(len(list(stats_glb.keys()))):
        stats_glb[list(stats_glb.keys())[j]] = {'mean':[],'std':[]}

    
    for i in range(len(list(spgram_dict.keys()))):
        aux = []
        for k in range(spgram_dict[list(spgram_dict.keys())[0]][0].shape[0]):
            if part == 'imag':
                aux.append(np.imag(spgram_dict[list(spgram_dict.keys())[i]][0][k,:,:].ravel()))
            elif part == 'abs':
                aux.append(np.abs(spgram_dict[list(spgram_dict.keys())[i]][0][k,:,:].ravel()))
            elif part == 'phase':
                aux.append(np.angle(spgram_dict[list(spgram_dict.keys())[i]][0][k,:,:].ravel()),False)
            else:
                aux.append(np.real(spgram_dict[list(spgram_dict.keys())[i]][0][k,:,:].ravel()))

        dict_metrics_aux = get_metrics(list_of_interest=aux)
        for j in range(len(list(stats_glb.keys()))):
            stats_glb[list(stats_glb.keys())[j]]['mean'].append(dict_metrics_aux[list(stats_glb.keys())[j]]['mean'])
            stats_glb[list(stats_glb.keys())[j]]['std'].append(dict_metrics_aux[list(stats_glb.keys())[j]]['std'])
    
    return stats_glb