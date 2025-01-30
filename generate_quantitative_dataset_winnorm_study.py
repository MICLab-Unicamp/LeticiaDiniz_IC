import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal.windows import hann, boxcar, flattop
from matplotlib.gridspec import GridSpec
import json
import math
import os
import utils
import functions_for_param_study as funcstud

##--------------------RECEIVE FILE WITH STUDY DESCRIPTION-----------------------------
config_path = input("Enter path for YAML file with winnorm study description: ")
config = utils.read_yaml(file=config_path)

##--------------------DEFINING STUDY VARIABLES----------------------------------------
qntty =int(config['fids'].get('qntty',100))
path_to_gt_fids =str(config['fids'].get('path_to_gt_fids','../sample_data.h5'))

name_of_study = str(config.get('name_of_study','winnorm_study'))
save_pictures_along_the_way = bool(config.get('save_pictures_along_the_way',False))

add_noise_to_fids = bool(config['amplitude_noise'].get('add_noise',False))
if add_noise_to_fids == True:
    noise_std_base = float(config['amplitude_noise']['noise_config'].get('std_base',6))
    noise_std_var = float(config['amplitude_noise']['noise_config'].get('std_var',2))
    noise_nmb_of_transients_to_combine = int(config['amplitude_noise']['noise_config'].get('nmb_of_transients_to_combine',160))

param_to_vary = config['study_parameters']['param_to_vary']
if param_to_vary != 'winnorm':
    raise Exception('This code only handles winnorm variation studies.')
try:
    win_name = list(config['study_parameters']['variation_details']['win'])
except KeyError:
    raise KeyError('You must specify study conditions: Missing definition of list with window types.')
try:
    norm_ = list(config['study_parameters']['variation_details']['norm'])
except KeyError:
    raise KeyError('You must specify study conditions: Missing definition of list with norms.')

for n in norm_:
    if n != 'abs' and n != 'm1p1' and n!= 'minmax' and n != 'zscore':
        raise Exception('Unknown normalization. Check function get_normalized_spectrogram in utils.py and addapt it to handle the desired normalization. Check also FWHM, peaks length and statistics calculation if norm is not zero centered.') 

hop_ = int(config['study_parameters']['fixed_params'].get('hop',8))
mfft_ = int(config['study_parameters']['fixed_params'].get('win','hann'))

window_ = []
for win in win_name:
    if win == 'hann':
        window_.append(hann(mfft_,sym=True))
    elif win == 'rect':
        window_.append(boxcar(mfft_,sym=True))
    elif win == 'flat':
        window_.append(flattop(mfft_,sym=True))
else:
    raise Exception('Unknown window type. Please check the script and addapt it to handle the desired window.')

perform_stats_analysis = bool(config['stats_analysis'].get('perform_stats_analysis',False))
if perform_stats_analysis == True:
    save_pictures_stats = bool(config['stats_analysis'].get('save_pictures',False))
    segmentation_values = {}
    for n in norm_:
        try: 
            segmentation_values[n] = list(config['stats_analysis']['segmentation_values'][n])
        except KeyError:
            raise KeyError('You must specify study conditions: Missing segmentation values for norm '+n+'.')

if save_pictures_along_the_way == True or save_pictures_stats == True:
    results_folder = './'+name_of_study+'/'
    os.makedirs(results_folder, exist_ok=True)

if add_noise_to_fids == True:
    print('Running winnorm study on '+str(qntty)+' noisy transients.')
else:
    print('Running winnorm study on '+str(qntty)+' GT transients.')
    

print('Winnorm variation: norms: '+str(norm_)+', wins: '+str(win_name)+'.'+
        '\nOther STFT parameters: \nhop: '+ str(hop_)+
         '\nmfft: '+str(mfft_)+
        '\nConsidering real part of STFT.')

if save_pictures_along_the_way == True or save_pictures_stats == True:
    print('WARNING: Figures configuration may depend on characteristics of the spectrogram, and therefore might not work for all cases. Keep this in mind while analysing data.')

##--------------------IMPORT GT FIDS--------------------------------------------------
if os.path.isfile(path_to_gt_fids) == False:
    raise Exception('Path to GT given is incorrect.')

try:
    with h5py.File(path_to_gt_fids) as hf:
        gt_fids = hf["ground_truth_fids"][()][:qntty]
        ppm = hf["ppm"][()][:qntty]
        t = hf["t"][()][:qntty]
except ImportError:
    raise ImportError("File with GT fids doesn't have the correct structure. Please check the script or the READ ME to understand the expect data structure.")

##--------------------GET Hz to ppm TRANSFORMATION PARAMETERS (a,b)-------------------
dwelltime = t[0,1]-t[0,0]
bandwidth = 1/dwelltime
N = gt_fids.shape[1]

#gts
spectra_gt_fids = np.fft.fftshift(np.fft.ifft(gt_fids,n=N,axis = 1), axes = 1)
spectra_gt_diff = spectra_gt_fids[:,:,1] - spectra_gt_fids[:,:,0]
freq = np.flip(np.fft.fftshift(np.fft.fftfreq(N, d = dwelltime)))

#to get ppm axis
idx_min = np.real(spectra_gt_diff[0,:]).argmin()
idx_max = np.real(spectra_gt_diff[0,:]).argmax()
#p = a*f + b
a = (ppm[0,idx_max] - ppm[0,idx_min])/(freq[idx_max]-freq[idx_min])
b = ppm[0,idx_max] - a*freq[idx_max]
#ppm_aux = b + freq*a

##--------------------CREATE NOISY TRANSIENTS IF DESIRED------------------------------
if add_noise_to_fids == True:
    print('Creating noisy transients by sampling '+str(qntty)+' Sigma_i values from uniform distribution ['+
            str(noise_std_base-noise_std_var)+','+std(noise_std_base+noise_std_var)+'). From each GT_i transient, we create '+
            str(noise_nmb_of_transients_to_combine)+' transients by adding amplitude noise from a normal distribuiton (0,Sigma_i). \n These '
            +str(noise_nmb_of_transients_to_combine)+' transients are then combined, so we get '+
            str(qntty)+' noisy fids to work with during the study.')
    print('This might take a while...')
    corrupted_fids = utils.create_corrupted_fids(gt=gt_fids,t=t,std_base=noise_std_base,std_var=noise_std_var,ntransients=noise_nmb_of_transients_to_combine)


##--------------------GENERATING GABA SPECTROGRAMS FROM FIDS FOR EVERY WINNORM COMB----
spgram_wn = {}
print('Creating GABA spectrograms for every win-norm combination and every fid signal. \nThis might take a while...')
for i in range(len(norm_)):
    spgram_wn['norm_'+norm_[i]] = {}
for i in range(len(norm_)):
    for j in range(len(window_)):
        if add_noise_to_fids == True:
            spgram, freq_spect, ppm_spect, t_spect = utils.get_normalized_spectrogram(fids=np.mean(corrupted_fids[:,:,1,:]-corrupted_fids[:,:,0,:],axis=2),bandwidth=bandwidth,window=window_[j],mfft=mfft_,hop=hop_,norm=norm_[i],correct_time=True,a=a,b=b)
        else:
            spgram, freq_spect, ppm_spect, t_spect = utils.get_normalized_spectrogram(fids=gt_fids[:,:,1]-gt_fids[:,:,0],bandwidth=bandwidth,window=window_[j],mfft=mfft_,hop=hop_,norm=norm_[i],correct_time=True,a=a,b=b)            
        spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]] = [spgram, freq_spect, ppm_spect, t_spect]

##--------------------EXTRACTING IMPORTANT SPECTROGRAM POINTS IN TIME AND FREQ--------
idx_time_0d05 = [] 
idx_time_0d4 = []
idx_time_0d6 = [] 
idx_freq_0ppm = [] 
idx_freq_1ppm = [] 
idx_freq_4ppm = [] 
idx_freq_8ppm = [] 
idx_freq_8d5ppm = []
idx_freq_NAA = [] 
idx_freq_GABA = []
idx_freq_Glx = []
for i in range(len(norm_)):
    list_of_t_spects = []
    list_of_ppm_spects = []
    for j in range(len(window_)):
        list_of_t_spects.append(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][-1])
        list_of_ppm_spects.append(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][2])
    idx_time_0d05.append(utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.05))
    idx_time_0d4.append(utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.4))
    idx_time_0d6.append(utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.6))
    idx_freq_0ppm.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=0))
    idx_freq_1ppm.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=1))
    idx_freq_4ppm.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=4))
    idx_freq_8ppm.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=8))
    idx_freq_8d5ppm.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=8.5))
    idx_freq_NAA.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=2.02))
    idx_freq_GABA.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=3.00))
    idx_freq_Glx.append(utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=3.75))

##--------MEASURING PEAK'S WIDTH IN SPECTROGRAM PROJECTION ON FREQUENCY AXIS: --------
print('Projecting spectrogram onto frequency axis to measure peaks full width at half maximum...')
fwhm_wn = {'NAA':[],'GABA':[],'Glx':[]}
fwhm_wn_real = {'NAA':[],'GABA':[],'Glx':[]}
#used for peak length measure
idx_fwhm_real = {}
for i in range(len(norm_)):
    list_projections_abs = []
    list_projections_real = []
    list_of_ppm_spects = []
    idx_fwhm_real['norm_'+norm_[i]] = {}
    for j in range(len(window_)):
        if norm_[i] == 'minmax':
            aux_mean_minmax = np.mean(np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0]),axis=(1,2),keepdims=True)
            aux_minmax = np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0]) - aux_mean_minmax
            aux_abs = np.sum(np.abs(aux_minmax),axis=2)
            aux_real = np.sum(aux_minmax,axis=2)
        else:
            aux_abs = np.sum(np.abs(np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0])),axis=2)
            aux_real = np.sum(np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0]),axis=2)
            
        list_projections_abs.append(aux_abs)
        list_projections_real.append(aux_real)
        list_of_ppm_spects.append(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][2])
        idx_fwhm_real['norm_'+norm_[i]]['window_'+win_name[j]] = {}
    fwhm_wn['NAA'].append([])
    fwhm_wn['NAA'][i], aux_idx_NAA_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_NAA[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=2.50,peak_ppm_minus=1.50,preference='positive')
    fwhm_wn['GABA'].append([])
    fwhm_wn['GABA'][i], aux_idx_GABA_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_GABA[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=3.50,peak_ppm_minus=2.50,preference='positive')
    fwhm_wn['Glx'].append([])
    fwhm_wn['Glx'][i], aux_idx_Glx_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_Glx[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=4.00,peak_ppm_minus=3.50,preference='positive')
    
    fwhm_wn_real['NAA'].append([])      
    fwhm_wn_real['NAA'][i], aux_idx_NAA_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_NAA[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=2.50,peak_ppm_minus=1.50,preference='negative')
    fwhm_wn_real['GABA'].append([])      
    fwhm_wn_real['GABA'][i], aux_idx_GABA_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_GABA[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=3.50,peak_ppm_minus=2.50,preference='positive')
    fwhm_wn_real['Glx'].append([])      
    fwhm_wn_real['Glx'][i], aux_idx_Glx_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_Glx[i],list_ppm=list_of_ppm_spects,peak_ppm_plus=4.00,peak_ppm_minus=3.50,preference='positive')
    
    #used for peak length measure
    for j in range(len(window_)):
        idx_fwhm_real['norm_'+norm_[i]]['window_'+win_name[j]]['NAA'] = aux_idx_NAA_real[j]
        idx_fwhm_real['norm_'+norm_[i]]['window_'+win_name[j]]['GABA'] = aux_idx_GABA_real[j]
        idx_fwhm_real['norm_'+norm_[i]]['window_'+win_name[j]]['Glx'] = aux_idx_Glx_real[j]

##--------------------MEASURING PEAKS ZERO CROSSING RATE FOR STRIPES PATTERN----------
print('Measuring zero crossing rate to capture important peaks stripes pattern...')
zcr_ = {'NAA':[],'GABA':[],'Glx':[]}
for i in range(len(norm_)):
    zcr_aux = funcstud.get_zcr_for_relevant_peaks_for_different_spgrams(spgram_dict=spgram_wn['norm_'+norm_[i]],idx_list_GABA=idx_freq_GABA[i],idx_list_NAA=idx_freq_NAA[i],idx_list_Glx=idx_freq_Glx[i],idx_time_list_0d4=idx_time_0d4[i])
    for key in list(zcr_.keys()):
        zcr_[key].append(zcr_aux[key])

##--------------------MEASURING PEAKS LENGHT------------------------------------------
print('Measuring peaks length...')
segm_dict = {}
sum_segment = {}
for i in range(len(norm_)):
    if norm_[i] == 'minmax':
        segm_dict['norm_'+norm_[i]] = funcstud.segment_relevant_peaks_dict(spgram_dict=spgram_wn['norm_'+norm_[i]],idx_list_1ppm=idx_freq_1ppm[i],idx_list_4ppm=idx_freq_4ppm[i],idx_list_GABA=idx_freq_GABA[i],idx_list_NAA=idx_freq_NAA[i],idx_list_Glx=idx_freq_Glx[i],idx_time_list_0d4=idx_time_0d4[i],idx_peaks_regions_limits_dict=idx_fwhm_real['norm_'+norm_[i]],not_zero_centered=True,center_value=0.5)
    else: 
        segm_dict['norm_'+norm_[i]] = funcstud.segment_relevant_peaks_dict(spgram_dict=spgram_wn['norm_'+norm_[i]],idx_list_1ppm=idx_freq_1ppm[i],idx_list_4ppm=idx_freq_4ppm[i],idx_list_GABA=idx_freq_GABA[i],idx_list_NAA=idx_freq_NAA[i],idx_list_Glx=idx_freq_Glx[i],idx_time_list_0d4=idx_time_0d4[i],idx_peaks_regions_limits_dict=idx_fwhm_real['norm_'+norm_[i]])
    sum_segment['norm_'+norm_[i]] = funcstud.get_length_relevant_peaks_for_different_spgrams(segm_dict=segm_dict['norm_'+norm_[i]],spgram_dict=spgram_wn['norm_'+norm_[i]],idx_list_1ppm=idx_freq_1ppm[i],idx_list_4ppm=idx_freq_4ppm[i],idx_list_GABA=idx_freq_GABA[i],idx_list_NAA=idx_freq_NAA[i],idx_list_Glx=idx_freq_Glx[i])

##--------------------VISUALIZATION OF SPECTROGRAM PROPERTIES-------------------------
if save_pictures_along_the_way == True:

    ##--------------------PIC OF SPECTROGRAMS-----------------------------------------
    print("Saving figure of concatenated spectrograms...")
    #concat
    concat_by_norm = {}
    for i in range(len(norm_)):
        concat_by_norm['norm_'+norm_[i]] = utils.concatenate_generic(selected_keys=list(spgram_wn['norm_'+norm_[i]].keys()),spgram_dict=spgram_wn['norm_'+norm_[i]],list_time_idx=idx_time_0d4[i],fid_idx_plot=0)
    #figure
    title_aux = ''
    for j in range(len(win_name)):
        if j < len(win_name)-1:
            title_aux=title_aux+win_name[j]+'/'
        else:
            title_aux=title_aux+win_name[j]
    fig,ax = plt.subplots(math.ceil(len(norm_)/2),2,figsize=(16,8))
    for i in range(len(norm_)):
        aux_m = np.mean(np.real(spgram_wn['norm_'+norm_[i]][0]))
        aux_s = np.std(np.real(spgram_wn['norm_'+norm_[i]][0]))
        im = ax.flat[i].imshow(np.real(concat_by_norm['norm_'+norm_[i]][idx_freq_1ppm[i][0]:idx_freq_4ppm[i][0],:]), origin='lower', aspect='auto',cmap='gray',vmin=aux_m-aux_s,vmax=aux_m+aux_s,
                extent = (0,concat_by_norm['norm_'+norm_[0]].shape[-1],np.flip(spgram_wn['norm_'+norm_[i]]['window_'+win_name[0]][2])[idx_freq_1ppm[i][0]],np.flip(spgram_wn['norm_'+norm_[i]]['window_'+win_name[0]][2])[idx_freq_4ppm[i][0]]))
        fig.colorbar(im, ax=ax.flat[i])
        ax.flat[i].set_xlabel('Pixels')
        ax.flat[i].set_ylabel('Chemical Shift [ppm]')
        ax.flat[i].set_title('Concat Spectrograms for norm: '+norm_[i]+'\nwin: '+title_aux)
    plt.savefig(results_folder+'concat_spgrams.png')
    plt.close()

    ##--------------------PICS OF SPECTROGRAM PROFILES------------------------------
    fig,ax = plt.subplots(math.ceil(len(norm_)/2),2,figsize=(16,8))
    for i in range(len(norm_)):
        ax.flat[i].plot(np.real(concat_by_norm['norm_'+norm_[i]][idx_freq_NAA[i][0],:]),color='b',label='NAA')
        ax.flat[i].plot(np.real(concat_by_norm['norm_'+norm_[i]][idx_freq_Glx[i][0],:]),color='m',label='GABA')
        ax.flat[i].plot(np.real(concat_by_norm['norm_'+norm_[i]][idx_freq_GABA[i][0],:]),color='orange',label='Glx')
        ax.flat[i].set_title('Peaks Profiles norm: '+norm_[i]+'\nwin: '+title_aux)
        ax.flat[i].set_xlabel('Columns')
        ax.flat[i].set_ylabel('Peaks profiles')
    plt.tight_layout()
    plt.savefig(results_folder+'peaks_profiles.png')
    plt.close()

    ##--------------------PICS OF SPECTROGRAM PROJECTION------------------------------
    print("Saving figures of sprectrogram's projections...")
    
    ##--------------------COMBINED PROJECTIONS ABS------------------------------------
    fig,ax = plt.subplots(len(norm_),1,figsize=(20,13))
    for i in range(len(norm_)):
        for j in range(len(window_)):
            aux = np.sum(np.abs(np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0][0,:,:])),axis=1)
            ax.flat[i].plot(np.flip(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][2]),aux,label='window = '+win_name[j])
            ax.flat[i].set_title('Proj(|Spectrogram|) for norm: '+norm_[i])
            ax.flat[i].set_xlim(4.2,1)  
            ax.flat[i].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(results_folder+'proj_abs_spgram_shifted.png')
    plt.close()

    ##--------------------COMBINED PROJECTIONS REAL------------------------------------
    fig,ax = plt.subplots(len(norm_),1,figsize=(20,13))
    for i in range(len(norm_)):
        for j in range(len(window_)):
            aux = np.sum(np.real(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][0][0,:,:]),axis=1)
            ax.flat[i].plot(np.flip(spgram_wn['norm_'+norm_[i]]['window_'+win_name[j]][2]),aux,label='window = '+win_name[j])
            ax.flat[i].set_title('Proj(|Spectrogram|) for norm: '+norm_[i])
            ax.flat[i].set_xlim(4.2,1)  
            ax.flat[i].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(results_folder+'proj_real_spgram_shifted.png')
    plt.close()

    ##--------------------FWHM MEASURED IN ABS/REAL PROJECTION------------------------------
    df_fwhm =  get_result_table(fwhm_wn,norm_,win_name)
    with open(results_folder+'fwhm_abs_proj.html', 'w') as f:
        f.write(df_fwhm.to_html())
    df_fwhm =  get_result_table(fwhm_wn_real,norm_,win_name)
    with open(results_folder+'fwhm_real_proj.html', 'w') as f:
        f.write(df_fwhm.to_html())

    ##---------------------ZCR---------------------------------------------------------
    print("Save ZCR figure...")
    df_zcr = get_result_table(zcr_,norm_,win_name)
    with open(results_folder+'zcr.html', 'w') as f:
        f.write(df_zcr.to_html())

    ##--------------------PEAKS LENGTH-------------------------------------------------
    print("Save peaks length figure...")
    sum_segment_df_form = {'NAA':[],'GABA':[],'Glx':[]}
    for i in range(len(norm_)):
        sum_segment_df_form['NAA'].append(sum_segment['norm_'+norm_[i]]['NAA'])
        sum_segment_df_form['GABA'].append(sum_segment['norm_'+norm_[i]]['GABA'])
        sum_segment_df_form['Glx'].append(sum_segment['norm_'+norm_[i]]['Glx'])
    df_length = get_result_table(sum_segment_df_form,norm_,win_name)
    with open(results_folder+'peaks_length.html', 'w') as f:
        f.write(df_length.to_html())

##--------------------STATS PROPERTIES------------------------------------------------
if perform_stats_analysis == True:
    print('Calculating statistical characteristics...')

    ##--------------------REGIONS MAXIMUM---------------------------------------------
    print('Calculating maximum absolute value for different ppm regions...')
    sections = [-4,-3,-2,-1,0,1,1.50,2.50,3.50,4,5,6,7,8.50,9,10]
    section_list = []
    for k in range(1,len(sections)):
        section_list.append(str(sections[k-1])+':'+str(sections[k]))
    section_list.append('residual')
    max_sec = {}
    for i in range(len(norm_)):
        max_sec['norm_'+norm_[i]] = {}
        for j in range(len(window_)):
            max_sec['norm_'+norm_[i]]['window_'+win_name[j]] = {'mean':[],'std':[]}

    for i in range(len(norm_)):
        idx_time_list_sup = []
        for j in range(len(window_)):
            idx_time_list_sup.append([idx_time_0d05[i][j],idx_time_0d05[i][j],idx_time_0d05[i][j],idx_time_0d05[i][j],idx_time_0d05[i][j],
                                    idx_time_0d4[i][j],idx_time_0d4[i][j],idx_time_0d4[i][j],idx_time_0d4[i][j],idx_time_0d4[i][j],
                                    idx_time_0d05[i][j],idx_time_0d05[i][j],
                                    idx_time_0d4[i][j],idx_time_0d05[i][j],idx_time_0d05[i][j]])
        max_sec_aux, mean_sec, std_sec = funcstud.get_max_mean_std_per_sections_for_different_spgrams(spgram_dict=spgram_wn['norm_'+norm_[i]], part='real', sections=sections, idx_time_list_sup=idx_time_list_sup)
        for j in range(len(window_)):
            for k in range(1,len(sections)):
                max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['mean'].append(max_sec_aux[str(sections[k-1])+':'+str(sections[k])]['mean'][j])
                max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['std'].append(max_sec_aux[str(sections[k-1])+':'+str(sections[k])]['std'][j])

    for i in range(len(norm_)):
        idx_time_list_inf = []
        for j in range(len(window_)):
            idx_time_list_inf.append([idx_time_0d6[i][j]])
        max_sec_aux, mean_sec, std_sec = funcstud.get_max_mean_std_per_sections_for_different_spgrams(spgram_dict=spgram_wn['norm_'+norm_[i]], part='real', sections=[1,4], idx_time_list_inf=idx_time_list_inf)
        for j in range(len(window_)):
            max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['mean'].append(max_sec_aux['1:4']['mean'][j])
            max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['std'].append(max_sec_aux['1:4']['std'][j])

    if save_pictures_stats == True:
        ##-------------------- FIGURE OF REGIONS MAXIMUM----------------------------
        plot_groups = [[0,1,2,3,4,5,11,13,14],[8,10,12],[7,9],[6],[-1]]
        colors = ['b','r','g']
        fig,ax = plt.subplots(len(norm_),5,figsize=(20,16))
        for i in range(len(norm_)):
            for j in range(len(window_)):
                aux = np.array(max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['mean'])
                aux_std = np.array(max_sec['norm_'+norm_[i]]['window_'+win_name[j]]['std'])
                for gr_id,group in enumerate(plot_groups):
                    for k_id,k in enumerate(group):
                        if k_id== 0:
                            ax.flat[5*i+gr_id].scatter(section_list[k],aux[k],label=' win: '+win_name[j],color=colors[j])
                        else:
                            ax.flat[5*i+gr_id].scatter(section_list[k],aux[k],color=colors[j])
                        ax.flat[5*i+gr_id].scatter(section_list[k],(aux+aux_std)[k],color=colors[j], marker='x')
                        ax.flat[5*i+gr_id].scatter(section_list[k],(aux-aux_std)[k],color=colors[j], marker='x')
                        ax.flat[5*i+gr_id].vlines(section_list[k],(aux-aux_std)[k],(aux+aux_std)[k],color=colors[j])
            for gr_id,group in enumerate(plot_groups):
                ax.flat[5*i+gr_id].legend(loc='center left',ncols=2)
                ax.flat[5*i+gr_id].set_title('PPM Regions Max Abs Values \nnorm: '+norm_[i])
                ax.flat[5*i+gr_id].set_ylabel('PPM Regions Max Abs Values')
                ax.flat[5*i+gr_id].set_xticks(range(len(group)))
                ax.flat[5*i+gr_id].tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.savefig(results_folder+'max_abs_value_of_ppm_regions.png')
        plt.close()

    ##--------------------SEGMENTATION BASED ON ABS VALUES------------------------
    if save_pictures_stats == True:
        ##-------------------- SEGMENTATION FIGURE--------------------------------
        for n in norm_:
            fig,ax = plt.subplots(len(win_name),len(segmentation_values)+1,figsize=(16,16))
            for j in range(len(window_)):
                if n == 'minmax':
                    aux_bin = np.abs(np.real(spgram_wn['norm_'+n]['window_'+win_name[j]][0][0,:,:])-0.5)
                else:
                    aux_bin = np.abs(np.real(spgram_wn['norm_'+n]['window_'+win_name[j]][0][0,:,:]))
                extent_seq = (spgram_wn['norm_'+n]['window_'+win_name[j]][-1][0],spgram_wn['norm_'+n]['window_'+win_name[j]][-1][-1],
                            np.flip(spgram_wn['norm_'+n]['window_'+win_name[j]][2])[0],np.flip(spgram_wn['norm_'+n]['window_'+win_name[j]][2])[-1])
                
                for idx_seg in range(len(segmentation_values)+1):
                    if idx_seg == 0:
                        seg_res = (aux_bin < segmentation_values[n][idx_seg]).astype('int')
                        ax.flat[(len(segmentation_values)+1)*j+idx_seg].set_title('win ='+win_name[j]+'\n |Spgram| < '+str(segmentation_values[idx_seg]))
                    elif idx_seg == len(segmentation_values):
                        seg_res = (aux_bin > segmentation_values[n][idx_seg-1]).astype('int')
                        ax.flat[(len(segmentation_values)+1)*j+idx_seg].set_title('win ='+win_name[j]+'\n |Spgram| > '+str(segmentation_values[idx_seg-1]))
                    else:
                        seg_res = (aux_bin > segmentation_values[n][idx_seg-1]).astype('int')*(aux_bin < segmentation_values[n][idx_seg]).astype('int')
                        ax.flat[(len(segmentation_values)+1)*j+idx_seg].set_title('win ='+win_name[j]+'\n'+str(segmentation_values[idx_seg-1])+' < |Spgram| < '+str(segmentation_values[idx_seg]))
                    
                    ax.flat[(len(segmentation_values)+1)*j+idx_seg].imshow(seg_res,cmap='gray',origin='lower',aspect='auto',
                                extent = extent_seq)
                    ax.flat[(len(segmentation_values)+1)*j+idx_seg].set_ylabel('Chemical Shift [ppm]')
                    ax.flat[(len(segmentation_values)+1)*j+idx_seg].set_xlabel('Time [s]')
            plt.tight_layout()
            plt.savefig(results_folder+'segmentation_visual_norm_'+n+'.png')
            plt.close()
                
    ##--------------------QNTTY OF PIXELS PER SEGMENTED REGION--------------------
    qntty_percent_regions = {}
    for i in range(len(norm_)):
        if norm_[i] == 'minmax':
            hist, bins, bins_centered, qntty_percent_regions['norm_'+norm_[i]] = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=aux,nbins=5000,part='real',regions=regions[norm_[i]],normalized=True,not_zero_centered=True,center_value=0.5)
        else:
            hist, bins, bins_centered, qntty_percent_regions['norm_'+norm_[i]] = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=aux,nbins=5000,part='real',regions=regions[norm_[i]],normalized=True)


    if save_pictures_stats == True:
        ##--------------------QNTTY OF PIXELS FIGURE------------------------------
        colors = ['b','r','g']
        fig,ax = plt.subplots(1,len(norm_),figsize=(20,4))
        for i in range(len(norm_)):
            reg_list  = list(qntty_percent_regions['norm_'+norm_[i]].keys())
            for reg_id,reg in enumerate(reg_list):
                    aux = np.array(qntty_percent_regions['norm_'+norm_[i]][reg]['mean'])
                    aux_std = np.array(qntty_percent_regions['norm_'+norm_[i]][reg]['std'])
                    for j in range(len(window_)):
                        if reg_id == 0:
                            ax.flat[i].scatter(reg,aux[j],label=' win: '+win_name[j],color=colors[j])
                        else:
                            ax.flat[i].scatter(reg,aux[j],color=colors[j])
                        ax.flat[i].scatter(reg,(aux+aux_std)[j],color=colors[j], marker='x')
                        ax.flat[i].scatter(reg,(aux-aux_std)[j],color=colors[j], marker='x')
                        ax.flat[i].vlines(reg,(aux-aux_std)[j],(aux+aux_std)[j],color=colors[j])
            ax.flat[i].legend(loc='upper right')
            ax.flat[i].set_title('Percentage of pixels x Segmentation interval \nnorm: '+norm_[i])
            ax.flat[i].set_xticks(range(len(reg_list)))
            ax.flat[i].set_xticklabels(reg_list, rotation=45)
            ax.flat[i].set_yscale('log')
            ax.flat[i].set_ylabel('Percentage of pixels')
        plt.tight_layout()
        plt.savefig(results_folder+'qntty_of_pixels_in_segm_region.png')
        plt.close()

    ##--------------------STATISTICS PER SEGMENTED REGION-------------------------
    stats_per_region = {}
    stats_global = {}
    for i in range(len(norm_)):
        if norm_[i] == 'minmax':
            stats_per_region['norm_'+norm_[i]] = funcstud.stats_per_segmented_regions_for_different_spgrams(regions_threshold=regions[norm_[i]],spgram_dict=spgram_wn['norm_'+norm_[i]], part='real', not_zero_centered=True, center_value=0.5)
        else:
            stats_per_region['norm_'+norm_[i]] = funcstud.stats_per_segmented_regions_for_different_spgrams(regions_threshold=regions[norm_[i]],spgram_dict=spgram_wn['norm_'+norm_[i]], part='real')
        stats_global['norm_'+norm_[i]] = utils.stats_global_for_different_spgrams(spgram_dict=spgram_wn['norm_'+norm_[i]], part='part')

    if save_pictures_stats == True:
        ##--------------------STATISTICS PER SEGM. REGION FIGURE------------------
        colors = ['b','r','g']
        fig,ax = plt.subplots(5,len(norm_),figsize=(20,30),sharex='col')
        stats_list = list(stats_per_region['norm_'+norm_[0]].keys())
        for sta_id,sta in enumerate(stats_list):
            for i in range(len(norm_)):
                reg_list  = list(stats_per_region['norm_'+norm_[i]][sta].keys())
                for reg_id in range(len(reg_list)+1):
                    if reg_id != len(reg_list):
                        reg = reg_list[reg_id]
                        aux = np.array(stats_per_region['norm_'+norm_[i]][sta][reg]['mean'])
                        aux_std = np.array(stats_per_region['norm_'+norm_[i]][sta][reg]['std'])
                    else:
                        reg = 'global'
                        aux = np.array(stats_global['norm_'+norm_[i]][sta]['mean'])
                        aux_std = np.array(stats_global['norm_'+norm_[i]][sta]['std'])
                    for j in range(len(window_)):
                        if reg_id == 0:
                            ax.flat[len(norm_)*sta_id+i].scatter(reg,aux[j],label=' win: '+win_name[j],color=colors[j])
                        else:
                            ax.flat[len(norm_)*sta_id+i].scatter(reg,aux[j],color=colors[j])
                        ax.flat[len(norm_)*sta_id+i].scatter(reg,(aux+aux_std)[j],color=colors[j], marker='x')
                        ax.flat[len(norm_)*sta_id+i].scatter(reg,(aux-aux_std)[j],color=colors[j], marker='x')
                        ax.flat[len(norm_)*sta_id+i].vlines(reg,(aux-aux_std)[j],(aux+aux_std)[j],color=colors[j])
                if i == 0:
                    ax.flat[len(norm_)*sta_id+i].legend(loc='upper right')
                ax.flat[len(norm_)*sta_id+i].set_title(sta+' norm: '+norm_[i])
                if sta_id == len(stats_list)-1:
                    ax.flat[len(norm_)*sta_id+i].set_xticks(range(len(reg_list)+1))
                    ax.flat[len(norm_)*sta_id+i].set_xticklabels(reg_list+['global'], rotation=45)
        plt.tight_layout()
        plt.savefig(results_folder+'stats_per_segm_region.png')
        plt.close()


##--------------------SAVING QUANTITATIVE METRICS DATA------------------------
print('Saving quantitative metrics in file...')
zcr_list = {}
zcr_list['NAA'] = [{},{},{},{}]
zcr_list['GABA'] = [{},{},{},{}]
zcr_list['Glx'] = [{},{},{},{}]
for ele in ['NAA','GABA','Glx']:
    for i in range(len(norm_)):
        zcr_list[ele][i]['mean'] = list(zcr_[ele][i]['mean'])
        zcr_list[ele][i]['std'] = list(zcr_[ele][i]['std'])

if save_pictures_along_the_way == True or save_pictures_stats == True:
    qnttive_file = results_folder+name_of_study+'.txt'
else:
    qnttive_file = name_of_study+'.txt'
with open(qnttive_file, "w") as f:
    f.write("fwhm_wn=")
    f.write(json.dumps(fwhm_wn))
    f.write('\n')
    f.write("fwhm_wn_real=")
    f.write(json.dumps(fwhm_wn_real))
    f.write('\n')
    f.write("sum_segment=")
    f.write(json.dumps(sum_segment))
    f.write('\n')
    f.write("zcr_=")
    f.write(json.dumps(zcr_list))
    f.write('\n')
    if perform_stats_analysis == True:
        f.write("max_sec=")
        f.write(json.dumps(max_sec))
        f.write('\n')
        f.write("qntty_percent_regions=")
        f.write(json.dumps(qntty_percent_regions))
        f.write('\n')
        f.write("stats_per_region=")
        f.write(json.dumps(stats_per_region))
        f.write('\n')
        f.write("stats_global=")
        f.write(json.dumps(stats_global))

print('All done!')