import numpy as np
import librosa
from utils import get_histogram, dict_with_stats,get_metrics
from utils import give_idx_ppm_point,give_idx_ppm_point_for_different_ppm_arrays




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

def histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict,nbins,part,regions,normalized, not_zero_centered=None, center_value=None):
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
    not_zero_centered, center_value (optional): if not_zero_centered is True than subtracts center_value of 
                    the spgram before considering the thresholds (thresholds are assumed to consider a center valued spgram)
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
        if not_zero_centered == True and center_value != None:
            hist_aux, bins_aux, bins_centered_aux = get_histogram(spgram=spgram_dict[list(spgram_dict.keys())[i]][0]-center_value,part=part,nbins=nbins,flatten=False,normalized=normalized)
        else:
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

