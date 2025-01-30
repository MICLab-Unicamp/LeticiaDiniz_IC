import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal.windows import hann, boxcar, flattop
from matplotlib.gridspec import GridSpec
import json
import os
import utils
import functions_for_param_study as funcstud

##--------------------RECEIVE FILE WITH STUDY DESCRIPTION-----------------------------
config_path = input("Enter path for YAML file with mfft study description: ")
config = utils.read_yaml(file=config_path)

##--------------------DEFINING STUDY VARIABLES----------------------------------------
qntty =int(config['fids'].get('qntty',100))
path_to_gt_fids =str(config['fids'].get('path_to_gt_fids','../sample_data.h5'))

name_of_study = str(config.get('name_of_study','mfft_study'))
save_pictures_along_the_way = bool(config.get('save_pictures_along_the_way',False))

add_noise_to_fids = bool(config['amplitude_noise'].get('add_noise',False))
if add_noise_to_fids == True:
    noise_std_base = float(config['amplitude_noise']['noise_config'].get('std_base',6))
    noise_std_var = float(config['amplitude_noise']['noise_config'].get('std_var',2))
    noise_nmb_of_transients_to_combine = int(config['amplitude_noise']['noise_config'].get('nmb_of_transients_to_combine',160))

param_to_vary = config['study_parameters']['param_to_vary']
if param_to_vary != 'mfft':
    raise Exception('This code only handles mfft variation studies.')
try:
    min_param = config['study_parameters']['variation_details']['min']
except KeyError:
    raise KeyError('You must specify study conditions: Missing minimum value for hop.')
try:
    max_param = config['study_parameters']['variation_details']['max']
except KeyError:
    raise KeyError('You must specify study conditions: Missing maximum value for hop.')
try:
    step_param = config['study_parameters']['variation_details']['step']
except KeyError:
    raise KeyError('You must specify study conditions: Missing step between hop values.')
mfft_ = np.arange(min_param,max_param,step_param).astype('int')

hop_ = int(config['study_parameters']['fixed_params'].get('hop',8))

win = str(config['study_parameters']['fixed_params'].get('win','hann'))
window_ = []
if win == 'hann':
    for i in range(mfft_.shape[0]):
        window_.append(hann(int(mfft_[i]),sym=True))
elif win == 'rect':
    for i in range(mfft_.shape[0]):
        window_.append(boxcar(int(mfft_[i]),sym=True))
elif win == 'flat':
    for i in range(mfft_.shape[0]):
        window_.append(flattop(int(mfft_[i]),sym=True))
else:
    raise Exception('Unknown window type. Please check the script and addapt it to handle the desired window.')

norm_ = str(config['study_parameters']['fixed_params'].get('norm','abs'))
if norm_ != 'abs' and norm_ != 'm1p1' and norm_ != 'minmax' and norm_ != 'zscore':
    raise Exception('Unknown normalization. Check function get_normalized_spectrogram in utils.py and addapt it to handle the desired normalization. Check also FWHM, peaks length and statistics calculation if norm is not zero centered.') 

perform_stats_analysis = bool(config['stats_analysis'].get('perform_stats_analysis',False))
if perform_stats_analysis == True:
    segmentation_values = list(config['stats_analysis']['segmentation_values'])
    save_pictures_stats = bool(config['stats_analysis'].get('save_pictures',False))

if save_pictures_along_the_way == True or save_pictures_stats == True:
    results_folder = './'+name_of_study+'/'
    os.makedirs(results_folder, exist_ok=True)

if add_noise_to_fids == True:
    print('Running mfft study on '+str(qntty)+' noisy transients.')
else:
    print('Running mfft study on '+str(qntty)+' GT transients.')
    
print('Mfft variation: ['+str(min_param)+','+str(max_param)+') stepping '+
        str(step_param)+' units. \nOther STFT parameters: \nhop: '+ str(hop_)+
         '\nwin: '+str(win)+ '\nnorm: '+str(norm_)+
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


##--------------------GENERATING GABA SPECTROGRAMS FROM FIDS FOR EVERY MFFT VALUE------
spgram_mfft = {}
print('Creating GABA spectrograms for every mfft value and every fid signal. \nThis might take a while...')
for i in range(mfft_.shape[0]):
    if add_noise_to_fids == True:
        spgram, freq_spect, ppm_spect, t_spect = utils.get_normalized_spectrogram(fids=np.mean(corrupted_fids[:,:,1,:]-corrupted_fids[:,:,0,:],axis=2),bandwidth=bandwidth,window=window_[i],mfft=int(mfft_[i]),hop=hop_,norm=norm_,correct_time=True,a=a,b=b)
    else:
        spgram, freq_spect, ppm_spect, t_spect = utils.get_normalized_spectrogram(fids=gt_fids[:,:,1]-gt_fids[:,:,0],bandwidth=bandwidth,window=window_[i],mfft=int(mfft_[i]),hop=hop_,norm=norm_,correct_time=True,a=a,b=b)
    spgram_mfft['mfft_'+str(mfft_[i])] = [spgram, freq_spect, ppm_spect, t_spect]

##--------------------EXTRACTING IMPORTANT SPECTROGRAM POINTS IN TIME AND FREQ--------
list_of_t_spects = []
list_of_ppm_spects = []
for i in range(mfft_.shape[0]):
    list_of_t_spects.append(spgram_mfft['mfft_'+str(mfft_[i])][-1])
    list_of_ppm_spects.append(spgram_mfft['mfft_'+str(mfft_[i])][2])
idx_time_0d05 = utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.05)
idx_time_0d4 = utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.4)
idx_time_0d6 = utils.give_idx_time_point_for_different_time_arrays(list_time_arrays=list_of_t_spects,time_point=0.6)
idx_freq_0ppm = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=0)
idx_freq_1ppm = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=1)
idx_freq_4ppm = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=4)
idx_freq_8ppm = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=8)
idx_freq_8d5ppm = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=8.5)
idx_freq_NAA = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=2.02)
idx_freq_GABA = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=3.00)
idx_freq_Glx = utils.give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_of_ppm_spects,ppm_point=3.75)

##--------MEASURING PEAK'S WIDTH IN SPECTROGRAM PROJECTION ON FREQUENCY AXIS: --------
print('Projecting spectrogram onto frequency axis to measure peaks full width at half maximum...')
list_projections_abs = []
list_projections_real = []
fwhm_mfft = {}
fwhm_mfft_real = {}
#used for peak length measure
idx_fwhm_real = {}
for i in range(len(mfft_)):
    if norm_ == 'minmax':
        aux_mean_minmax = np.mean(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0]),axis=(1,2),keepdims=True)
        aux_minmax = np.real(spgram_mfft['mfft_'+str(mfft_[i])][0]) - aux_mean_minmax
        list_projections_abs.np.sum(np.abs(aux_minmax),axis=2)
        list_projections_abs.np.sum(aux_minmax,axis=2)
    else:
        list_projections_abs.append(np.sum(np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0])), axis = 2))
        list_projections_real.append(np.sum(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0]), axis = 2))
    idx_fwhm_real['mfft_'+str(mfft_[i])] = {}

fwhm_mfft['NAA'], aux_idx_NAA_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_NAA,list_ppm=list_of_ppm_spects,peak_ppm_plus=2.50,peak_ppm_minus=1.50,preference='positive')
fwhm_mfft['GABA'], aux_idx_GABA_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_GABA,list_ppm=list_of_ppm_spects,peak_ppm_plus=3.50,peak_ppm_minus=2.50,preference='positive')
fwhm_mfft['Glx'], aux_idx_Glx_abs = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_abs,list_peak_idx=idx_freq_Glx,list_ppm=list_of_ppm_spects,peak_ppm_plus=4.00,peak_ppm_minus=3.50,preference='positive')

fwhm_mfft_real['NAA'], aux_idx_NAA_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_NAA,list_ppm=list_of_ppm_spects,peak_ppm_plus=2.50,peak_ppm_minus=1.50,preference='negative')
fwhm_mfft_real['GABA'], aux_idx_GABA_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_GABA,list_ppm=list_of_ppm_spects,peak_ppm_plus=3.50,peak_ppm_minus=2.50,preference='positive')
fwhm_mfft_real['Glx'], aux_idx_Glx_real = utils.get_fwhm_in_ppm_for_different_signals(list_signals=list_projections_real,list_peak_idx=idx_freq_Glx,list_ppm=list_of_ppm_spects,peak_ppm_plus=4.00,peak_ppm_minus=3.50,preference='positive')

#used for peak length measure
for i in range(len(mfft_)):
    idx_fwhm_real['mfft_'+str(mfft_[i])]['NAA'] = aux_idx_NAA_real[i]
    idx_fwhm_real['mfft_'+str(mfft_[i])]['GABA'] = aux_idx_GABA_real[i]
    idx_fwhm_real['mfft_'+str(mfft_[i])]['Glx'] = aux_idx_Glx_real[i]

##--------------------MEASURING PEAKS ZERO CROSSING RATE FOR STRIPES PATTERN----------
print('Measuring zero crossing rate to capture important peaks stripes pattern...')
zcr_ = funcstud.get_zcr_for_relevant_peaks_for_different_spgrams(spgram_dict=spgram_mfft,idx_list_GABA=idx_freq_GABA,idx_list_NAA=idx_freq_NAA,idx_list_Glx=idx_freq_Glx,idx_time_list_0d4=idx_time_0d4)

##--------------------MEASURING PEAKS LENGHT------------------------------------------
print('Measuring peaks length to capture influence of mfft in the horizontal direction...')
if norm_ == 'minmax':
    segm_dict = funcstud.segment_relevant_peaks_dict(spgram_dict=spgram_mfft,idx_list_1ppm=idx_freq_1ppm,idx_list_4ppm=idx_freq_4ppm,idx_list_GABA=idx_freq_GABA,idx_list_NAA=idx_freq_NAA,idx_list_Glx=idx_freq_Glx,idx_time_list_0d4=idx_time_0d4,idx_peaks_regions_limits_dict=idx_fwhm_real,not_zero_centered=True,center_value=0.5)
else:
    segm_dict = funcstud.segment_relevant_peaks_dict(spgram_dict=spgram_mfft,idx_list_1ppm=idx_freq_1ppm,idx_list_4ppm=idx_freq_4ppm,idx_list_GABA=idx_freq_GABA,idx_list_NAA=idx_freq_NAA,idx_list_Glx=idx_freq_Glx,idx_time_list_0d4=idx_time_0d4,idx_peaks_regions_limits_dict=idx_fwhm_real)
sum_segment = funcstud.get_length_relevant_peaks_for_different_spgrams(segm_dict=segm_dict,spgram_dict=spgram_mfft,idx_list_1ppm=idx_freq_1ppm,idx_list_4ppm=idx_freq_4ppm,idx_list_GABA=idx_freq_GABA,idx_list_NAA=idx_freq_NAA,idx_list_Glx=idx_freq_Glx)

##--------------------VISUALIZATION OF SPECTROGRAM PROPERTIES-------------------------
if save_pictures_along_the_way == True:

    ##--------------------PIC OF SPECTROGRAMS-----------------------------------------
    print("Saving figure of concatenated spectrograms...")
    #preparation
    aux_concat = (np.arange(0,len(mfft_),int(len(mfft_)/6))).tolist()
    plot_concat = []
    for idx in aux_concat:
        plot_concat.append(mfft_[idx])
    #concat
    spgram_mfft_concat = funcstud.concatenate_different_mfft(list_mfft_idx=plot_concat,spgram_dict=spgram_mfft,time_idx=idx_time_0d4[0],fid_idx_plot=0)
    #figure
    fig,ax = plt.subplots(1,2,figsize=(16,4))
    im = ax.flat[0].imshow(np.real(spgram_mfft_concat), origin='lower',cmap='gray',aspect='auto',vmin=-0.04,vmax=0.04)
    count = 0
    center = spgram_mfft_concat.shape[0]/2
    idx_aux = idx_time_0d4[0]
    for i in range(0,len(mfft_),int(len(mfft_)/6)):
        aux = spgram_mfft['mfft_'+str(mfft_[i])][0].shape[1]
        ax.flat[0].hlines(center-int(aux/2)+idx_freq_1ppm[i],count,count+idx_aux-1)
        ax.flat[0].hlines(center-int(aux/2)+idx_freq_4ppm[i],count,count+idx_aux-1)
        count = count+idx_aux
    ax.flat[0].set_title('Concat Spectrograms with mfft $\in$ ['+str(mfft_[0])+','+str(mfft_[-1])+') \n with step '+str(int(len(mfft_)/6)))
    ax.flat[0].set_xlabel('Pixels')
    ax.flat[0].set_ylabel('Pixels')
    fig.colorbar(im, ax = ax[0])
    aux = spgram_mfft['mfft_'+str(mfft_[2])][0].shape[1]
    im = ax.flat[1].imshow(np.real(spgram_mfft_concat[int(center)-int(aux/2)+idx_freq_1ppm[2]:int(center)-int(aux/2)+idx_freq_4ppm[2],:]), origin='lower', 
                aspect='auto',cmap='gray',vmin=-0.01,vmax=0.01)
    ax.flat[1].set_title('Zoom in GABA Peak')
    ax.flat[1].set_xlabel('Pixels')
    ax.flat[1].set_ylabel('Pixels')
    fig.colorbar(im, ax = ax[1])
    plt.savefig(results_folder+'concat_spgrams.png')
    plt.close()

    ##--------------------PICS OF SPECTROGRAM PROJECTION------------------------------
    print("Saving figures of sprectrogram's projections...")
    
    ##--------------------COMBINED PROJECTIONS ABS------------------------------------
    plot_id = np.arange(0,len(mfft_),int(len(mfft_)/6))
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    for i in range(len(plot_id)):
        aux = (np.sum(np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][0][0,:,:])),axis=1))
        ax[0].plot(np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][2]),aux+2*i,label='mfft = '+str(mfft_[plot_id[i]]))
        ax[1].plot(np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][2]),aux+0.25*i,label='mfft = '+str(mfft_[plot_id[i]]))
    ax[0].set_xlim(3.5,1)  
    ax[1].set_xlim(4.2,3.43)
    ax[1].set_ylim(-0.1,5)  
    ax[0].legend(loc='upper left',ncols=2)
    ax[1].legend(loc='upper left')
    ax[0].set_title('GABA and NAA peaks in Proj(|Spectrogram|)')
    ax[1].set_title('Glx peak in Proj(|Spectrogram|)')
    ax[0].set_xlabel('Chemical Shift [ppm]')
    ax[1].set_xlabel('Chemical Shift [ppm]')
    ax[0].set_ylabel('Vertically Shifted Projections')
    ax[1].set_ylabel('Vertically Shifted Projections')
    plt.tight_layout()
    plt.savefig(results_folder+'proj_abs_spgram_shifted.png')
    plt.close()

    ##--------------------COMBINED PROJECTIONS REAL------------------------------------
    plot_id = np.arange(0,len(mfft_),int(len(mfft_)/6))
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    for i in range(len(plot_id)):
        aux = (np.sum(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][0][0,:,:]),axis=1))    
        ax[0].plot(np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][2]),aux+2*i,label='mfft = '+str(mfft_[plot_id[i]]))
        ax[1].plot(np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[i]])][2]),aux+0.25*i,label='mfft = '+str(mfft_[plot_id[i]]))
    ax[0].set_xlim(3.5,1)  
    ax[1].set_xlim(4.2,3.43)
    ax[1].set_ylim(-0.1,5)  
    ax[0].legend(loc='upper left',ncols=2)
    ax[1].legend(loc='upper left')
    ax[0].set_title('GABA and NAA peaks in Proj(Spectrogram)')
    ax[1].set_title('Glx peak in Proj(Spectrogram)')
    ax[0].set_xlabel('Chemical Shift [ppm]')
    ax[1].set_xlabel('Chemical Shift [ppm]')
    ax[0].set_ylabel('Vertically Shifted Projections')
    ax[1].set_ylabel('Vertically Shifted Projections')
    plt.tight_layout()
    plt.savefig(results_folder+'proj_real_spgram_shifted.png')
    plt.close()

    ##--------------------FWHM MEASURED IN ABS/REAL PROJECTION------------------------------
    fig,ax = plt.subplots(1,2,figsize=(15,3))
    ax[0].plot(mfft_,fwhm_mfft['NAA']['mean'],label='NAA',color='b')
    ax[0].fill_between(mfft_, np.array(fwhm_mfft['NAA']['mean']) - np.array(fwhm_mfft['NAA']['std']), 
                            np.array(fwhm_mfft['NAA']['mean']) + np.array(fwhm_mfft['NAA']['std']), alpha=0.35, color = 'b')
    ax[0].plot(mfft_,fwhm_mfft['GABA']['mean'],label='GABA',color='orange')
    ax[0].fill_between(mfft_, np.array(fwhm_mfft['GABA']['mean']) - np.array(fwhm_mfft['GABA']['std']), 
                            np.array(fwhm_mfft['GABA']['mean']) + np.array(fwhm_mfft['GABA']['std']), alpha=0.35, color = 'orange')
    ax[0].plot(mfft_,fwhm_mfft['Glx']['mean'],label='Glx',color='g')
    ax[0].fill_between(mfft_, np.array(fwhm_mfft['Glx']['mean']) - np.array(fwhm_mfft['Glx']['std']), 
                            np.array(fwhm_mfft['Glx']['mean']) + np.array(fwhm_mfft['Glx']['std']), alpha=0.35, color = 'g')
    ax[0].set_title('FWHM in Proj(|Spectrogram|) x mfft')
    ax[0].set_xlabel('mfft')
    ax[0].set_ylabel('FWHM in Proj(|Spectrogram|)')
    ax[0].legend(loc='upper left')
    ax[1].plot(mfft_,fwhm_mfft_real['NAA']['mean'],label='NAA',color='b')
    ax[1].fill_between(mfft_, np.array(fwhm_mfft_real['NAA']['mean']) - np.array(fwhm_mfft_real['NAA']['std']), 
                            np.array(fwhm_mfft_real['NAA']['mean']) + np.array(fwhm_mfft_real['NAA']['std']), alpha=0.35, color = 'b')
    ax[1].plot(mfft_,fwhm_mfft_real['GABA']['mean'],label='GABA',color='orange')
    ax[1].fill_between(mfft_, np.array(fwhm_mfft_real['GABA']['mean']) - np.array(fwhm_mfft_real['GABA']['std']), 
                            np.array(fwhm_mfft_real['GABA']['mean']) + np.array(fwhm_mfft_real['GABA']['std']), alpha=0.35, color = 'orange')
    ax[1].plot(mfft_,fwhm_mfft_real['Glx']['mean'],label='Glx',color='g')
    ax[1].fill_between(mfft_, np.array(fwhm_mfft_real['Glx']['mean']) - np.array(fwhm_mfft_real['Glx']['std']), 
                            np.array(fwhm_mfft_real['Glx']['mean']) + np.array(fwhm_mfft_real['Glx']['std']), alpha=0.35, color = 'g')
    ax[1].set_title('FWHM in Proj(Spectrogram) x mfft')
    ax[1].set_xlabel('mfft')
    ax[1].set_ylabel('FWHM in Proj(Spectrogram)')
    ax[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(results_folder+'fwhm_abs_and_real_projs.png')
    plt.close()

    ##--------------------ZCR AND PEAKS PROFILES---------------------------------------
    print("Save ZCR figure...")
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(2,3,height_ratios=[1,2])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    ax1.plot(mfft_,zcr_['NAA']['mean'],color='b',label='NAA')
    ax1.fill_between(mfft_, np.array(zcr_['NAA']['mean']) - np.array(zcr_['NAA']['std']), 
                            np.array(zcr_['NAA']['mean']) + np.array(zcr_['NAA']['std']), alpha=0.35, color = 'b')
    ax1.plot(mfft_,zcr_['GABA']['mean'],color='r',label='GABA')
    ax1.fill_between(mfft_, np.array(zcr_['GABA']['mean']) - np.array(zcr_['GABA']['std']), 
                            np.array(zcr_['GABA']['mean']) + np.array(zcr_['GABA']['std']), alpha=0.35, color = 'r')
    ax1.plot(mfft_,zcr_['Glx']['mean'],color='g',label='Glx')
    ax1.fill_between(mfft_, np.array(zcr_['Glx']['mean']) - np.array(zcr_['Glx']['std']), 
                            np.array(zcr_['Glx']['mean']) + np.array(zcr_['Glx']['std']), alpha=0.35, color = 'g')
    ax1.legend(loc='upper right')
    ax1.set_title('ZCR for peak lines x mfft')
    ax1.set_ylabel('ZCR')
    ax1.set_xlabel('mfft')

    for i in range(0,len(mfft_),int(len(mfft_)/6)):
        ax2.plot(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][0,idx_freq_NAA[i],:idx_time_0d4[i]])-np.mean(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][:,idx_freq_NAA[i],:idx_time_0d4[i]]),axis=1)[0])
        ax3.plot(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][0,idx_freq_GABA[i],:idx_time_0d4[i]])-np.mean(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][:,idx_freq_GABA[i],:idx_time_0d4[i]]),axis=1)[0])
        ax4.plot(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][0,idx_freq_Glx[i],:idx_time_0d4[i]])-np.mean(np.real(spgram_mfft['mfft_'+str(mfft_[i])][0][:,idx_freq_Glx[i],:idx_time_0d4[i]]),axis=1)[0])
    ax2.set_title('NAA Peak till 0.4s center in 0')
    ax2.set_ylabel('NAA Profile')
    ax2.set_xlabel('Columns')
    ax3.set_title('GABA Peak till 0.4s center in 0')
    ax3.set_ylabel('GABA Profile')
    ax3.set_xlabel('Columns')
    ax4.set_title('Glx Peak till 0.4s center in 0')
    ax4.set_ylabel('Glx Profile')
    ax4.set_xlabel('Columns')
    plt.tight_layout()
    plt.savefig(results_folder+'zcr_and_peaks_profiles.png')
    plt.close()

    ##--------------------PEAKS LENGTH-------------------------------------------------
    print("Save peaks length figure...")
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(mfft_,sum_segment['NAA']['mean'],label='NAA',color='b')
    ax.fill_between(mfft_, np.array(sum_segment['NAA']['mean']) - np.array(sum_segment['NAA']['std']), 
                            np.array(sum_segment['NAA']['mean']) + np.array(sum_segment['NAA']['std']), alpha=0.35, color = 'b')
    ax.plot(mfft_,sum_segment['GABA']['mean'],label='GABA',color='r')
    ax.fill_between(mfft_, np.array(sum_segment['GABA']['mean']) - np.array(sum_segment['GABA']['std']), 
                            np.array(sum_segment['GABA']['mean']) + np.array(sum_segment['GABA']['std']), alpha=0.35, color = 'r')
    ax.plot(mfft_,sum_segment['Glx']['mean'],label='Glx',color='g')
    ax.fill_between(mfft_, np.array(sum_segment['Glx']['mean']) - np.array(sum_segment['Glx']['std']), 
                            np.array(sum_segment['Glx']['mean']) + np.array(sum_segment['Glx']['std']), alpha=0.35, color = 'g')
    ax.set_xlabel('mfft')
    ax.set_ylabel('Peaks Length')
    ax.set_title('Peaks Length')
    ax.legend(loc='upper left',ncols=2)
    plt.tight_layout()
    plt.savefig(results_folder+'peaks_length.png')
    plt.close()


##--------------------STATS PROPERTIES------------------------------------------------
if perform_stats_analysis == True:
    print('Calculating statistical characteristics...')

    ##--------------------REGIONS MAXIMUM---------------------------------------------
    print('Calculating maximum absolute value for different ppm regions...')
    sections = [-4,-3,-2,-1,0,1,1.50,2.50,3.50,4,5,6,7,8.50,9,10]
    idx_time_list_sup = []
    for i in range(len(mfft_)):
        idx_time_list_sup.append([idx_time_0d05[i],idx_time_0d05[i],idx_time_0d05[i],idx_time_0d05[i],idx_time_0d05[i],
                                idx_time_0d4[i],idx_time_0d4[i],idx_time_0d4[i],idx_time_0d4[i],idx_time_0d4[i],
                                idx_time_0d05[i],idx_time_0d05[i],
                                idx_time_0d4[i],idx_time_0d05[i],idx_time_0d05[i]])
    max_sec, mean_sec, std_sec = funcstud.get_max_mean_std_per_sections_for_different_spgrams(spgram_dict=spgram_mfft, part='real', sections=sections, idx_time_list_sup=idx_time_list_sup)

    idx_time_list_inf = []
    for i in range(len(mfft_)):
        idx_time_list_inf.append([idx_time_0d6[i]])
    max_sec_aux, mean_sec_aux, std_sec_aux = funcstud.get_max_mean_std_per_sections_for_different_spgrams(spgram_dict=spgram_mfft, part='real', sections=[1,4], idx_time_list_inf=idx_time_list_inf)
    max_sec['residual'] = max_sec_aux['1:4']
    mean_sec['residual'] = mean_sec_aux['1:4']
    std_sec['residual'] = std_sec_aux['1:4']

    if save_pictures_stats == True:
        ##-------------------- FIGURE OF REGIONS MAXIMUM----------------------------
        fig,ax = plt.subplots(2,3,figsize=(16,8))
        sections = [-4,-3,-2,-1,0,1,1.50,2.50,3.50,4,5,6,7,8.50,9,10]

        first_plot = [0,1,2,3,4,5,11,13,14]
        second_plot = [8,10,12]
        third_plot = [7,9]
        fourth_plot = [6]
        plot_seq = [first_plot,second_plot,third_plot,fourth_plot]
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

        for i in range(len(plot_seq)):
            for j in range(len(plot_seq[i])):
                aux = max_sec[str(sections[plot_seq[i][j]])+':'+str(sections[plot_seq[i][j]+1])]['mean']
                aux_std = max_sec[str(sections[plot_seq[i][j]])+':'+str(sections[plot_seq[i][j]+1])]['std']
                ax.flat[i].plot(mfft_,aux,label =str(sections[plot_seq[i][j]])+':'+str(sections[plot_seq[i][j]+1]),marker='o',color=colors[j])
                ax.flat[i].fill_between(mfft_, np.array(aux) - np.array(aux_std), 
                                np.array(aux) + np.array(aux_std), alpha=0.35, color=colors[j])

        ax.flat[4].plot(mfft_,max_sec['residual']['mean'],label='residual',marker='o')
        ax.flat[4].fill_between(mfft_, np.array(max_sec['residual']['mean']) - np.array(max_sec['residual']['std']), 
                                np.array(max_sec['residual']['mean']) + np.array(max_sec['residual']['std']), alpha=0.35, color=colors[j])
        for i in range(5):
            ax.flat[i].legend(loc='upper right',ncols=3)
            ax.flat[i].set_title('PPM Regions Max Abs Values x mfft')
            ax.flat[i].set_xlabel('mfft')
            ax.flat[i].set_ylabel('PPM Region Max Abs Values')
            
        ax.flat[5].imshow(np.real(spgram_mfft['mfft_'+str(mfft_[45])][0][0,:,:idx_time_0d4[45]]), origin='lower',cmap='gray',aspect='auto',vmin=-0.04,vmax=0.04,
            extent = (spgram_mfft['mfft_'+str(mfft_[45])][-1][0],spgram_mfft['mfft_'+str(mfft_[45])][-1][idx_time_0d4[45]],
                     np.flip(spgram_mfft['mfft_'+str(mfft_[45])][2])[0],np.flip(spgram_mfft['mfft_'+str(mfft_[45])][2])[-1]))
        ax.flat[5].set_title('Example of Spectrogram for \n ppm region reference')
        ax.flat[5].set_xlabel('Time [s]')
        ax.flat[5].set_ylabel('Chemical Shift [ppm]')
        plt.tight_layout()
        plt.savefig(results_folder+'max_abs_value_of_ppm_regions.png')
        plt.close()

    ##--------------------SEGMENTATION BASED ON ABS VALUES------------------------
    if save_pictures_stats == True:
        ##-------------------- SEGMENTATION FIGURE--------------------------------
        plot_id = np.arange(0,len(mfft_),int(len(mfft_)/6))
        fig,ax = plt.subplots(len(plot_id),len(segmentation_values)+1,figsize=(16,25))
        for idx in range(len(plot_id)):
            for idx_seg in range(len(segmentation_values)+1):
                if idx_seg == 0:
                    seg_res = (np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][0][0,:,:])) < segmentation_values[idx_seg]).astype('int')
                    ax.flat[(len(segmentation_values)+1)*idx+idx_seg].set_title('mfft ='+str(mfft_[plot_id[idx]])+'\n |Spgram| < '+str(segmentation_values[idx_seg]))
                elif idx_seg == len(segmentation_values):
                    seg_res = (np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][0][0,:,:])) > segmentation_values[idx_seg-1]).astype('int')
                    ax.flat[(len(segmentation_values)+1)*idx+idx_seg].set_title('mfft ='+str(mfft_[plot_id[idx]])+'\n |Spgram| > '+str(segmentation_values[idx_seg-1]))
                else:
                    seg_res = (np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][0][0,:,:])) > segmentation_values[idx_seg-1]).astype('int')*(np.abs(np.real(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][0][0,:,:])) < segmentation_values[idx_seg]).astype('int')
                    ax.flat[(len(segmentation_values)+1)*idx+idx_seg].set_title('mfft ='+str(mfft_[plot_id[idx]])+'\n'+str(segmentation_values[idx_seg-1])+' < |Spgram| < '+str(segmentation_values[idx_seg]))

                ax.flat[(len(segmentation_values)+1)*idx+idx_seg].imshow(seg_res,cmap='gray',origin='lower',aspect='auto',
                                extent = (spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][-1][0],spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][-1][-1],
                                np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][2])[0],np.flip(spgram_mfft['mfft_'+str(mfft_[plot_id[idx]])][2])[-1]))
                ax.flat[(len(segmentation_values)+1)*idx+idx_seg].set_ylabel('Chemical Shift [ppm]')
                ax.flat[(len(segmentation_values)+1)*idx+idx_seg].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(results_folder+'segmentation_visual.png')
        plt.close()

    ##--------------------QNTTY OF PIXELS PER SEGMENTED REGION x MFFT-------------
    if norm_ == 'minmax':
        hist, bins, bins_centered, qntty_percent_regions = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=spgram_mfft,nbins=5000,part='real',regions=segmentation_values,normalized=True,not_zero_centered=True,center_value=0.5)
        hist_absolute, bins_absolute, bins_centered_absolute, qntty_absolute_regions = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=spgram_mfft,nbins=5000,part='real',regions=segmentation_values,normalized=False,not_zero_centered=True,center_value=0.5)
    else:
        hist, bins, bins_centered, qntty_percent_regions = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=spgram_mfft,nbins=5000,part='real',regions=segmentation_values,normalized=True)
        hist_absolute, bins_absolute, bins_centered_absolute, qntty_absolute_regions = funcstud.histogram_for_different_spgram_and_qntty_per_histogram_region(spgram_dict=spgram_mfft,nbins=5000,part='real',regions=segmentation_values,normalized=False)

    if save_pictures_stats == True:
        ##--------------------QNTTY OF PIXELS FIGURE------------------------------
        colors = ['b','r','g','m','orange','c']
        fig,ax = plt.subplots(2,len(segmentation_values)+1,figsize=(20,8))
        for i in range(len(list(qntty_percent_regions.keys()))):
            aux = np.array(qntty_percent_regions[list(qntty_percent_regions.keys())[i]]['mean'])
            aux_std = np.array(qntty_percent_regions[list(qntty_percent_regions.keys())[i]]['std'])

            aux_absolute = np.array(qntty_absolute_regions[list(qntty_absolute_regions.keys())[i]]['mean'])
            aux_absolute_std = np.array(qntty_absolute_regions[list(qntty_absolute_regions.keys())[i]]['std'])
            ax[0,i].plot(mfft_,aux,marker='o',color=colors[i])
            ax[0,i].fill_between(mfft_, aux - aux_std, 
                            aux + aux_std, alpha=0.35, color = colors[i])
            ax[0,i].set_title('Percentage of pixels in \n Segmentation interval:\n'+list(qntty_percent_regions.keys())[i])
            ax[0,i].set_xlabel('mfft')
            ax[0,i].set_ylabel('Percentage of pixels')
            ax[1,i].plot(mfft_,aux_absolute,marker='o',color=colors[i])
            ax[1,i].fill_between(mfft_, aux_absolute - aux_absolute_std, 
                            aux_absolute + aux_absolute_std, alpha=0.35, color = colors[i])
            ax[1,i].set_title('Nmb of pixels in \n Segmentation interval:\n'+list(qntty_absolute_regions.keys())[i])
            ax[1,i].set_xlabel('mfft')
            ax[1,i].set_ylabel('Nmb of pixels')
        plt.tight_layout()
        plt.savefig(results_folder+'qntty_of_pixels_in_segm_region.png')
        plt.close()

    ##--------------------STATISTICS PER SEGMENTED REGION x MFFT------------------
    if norm_ == 'minmax':
        stats_per_region = funcstud.stats_per_segmented_regions_for_different_spgrams(regions_threshold=segmentation_values,spgram_dict=spgram_mfft,part='real',not_zero_centered=True,center_value=0.5)
        stats_global = utils.stats_global_for_different_spgrams(spgram_dict=spgram_mfft,part='part',not_zero_centered=True,center_value=0.5)
    else:
        stats_per_region = funcstud.stats_per_segmented_regions_for_different_spgrams(regions_threshold=segmentation_values,spgram_dict=spgram_mfft,part='real')
        stats_global = utils.stats_global_for_different_spgrams(spgram_dict=spgram_mfft,part='part')

    if save_pictures_stats == True:
        ##--------------------STATISTICS PER SEGM. REGION FIGURE------------------
        nrows = len(segmentation_values)+2
        ratios = []
        for i in range(nrows-1):
            ratios.append(1)
        ratios.append(2)
        fig,ax = plt.subplots(nrows,5,figsize=(20,15), sharex='col', height_ratios=ratios)
        stat_name = ['mean','std','median','skewness','kurtosis']
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#17becf']
        for j in range(len(stat_name)):
            for i in range(len(list(stats_per_region[stat_name[j]].keys()))):
                aux = np.array(stats_per_region[stat_name[j]][list(stats_per_region[stat_name[j]].keys())[i]]['mean'])
                aux_std = np.array(stats_per_region[stat_name[j]][list(stats_per_region[stat_name[j]].keys())[i]]['std'])
                ax[i,j].plot(mfft_,aux,marker='o',color=colors[i],label=list(stats_per_region[stat_name[j]].keys())[i])
                ax[i,j].fill_between(mfft_, aux - aux_std, 
                                aux + aux_std, alpha=0.35, color = colors[i])
                if i == 0:
                    ax[i,j].set_title(stat_name[j]+' x mfft')
                if j == 0:
                    ax[i,j].set_ylabel('Segm. Interval: '+list(stats_per_region[stat_name[j]].keys())[i])

        colors = ['#bcbd22']
        for i in range(len(stat_name)):
            aux = np.array(stats_global[stat_name[i]]['mean'])
            aux_std = np.array(stats_global[stat_name[i]]['std'])
            ax[nrows-1,i].plot(mfft_,aux,marker='o',color=colors[0],label="global")
            ax[nrows-1,i].fill_between(mfft_, aux - aux_std, 
                            aux + aux_std, alpha=0.35, color = colors[0])
            if i == 0:
                ax[nrows-1,i].set_ylabel('Global Spgram')
        for i in range(5):
            ax[nrows-1,i].set_xlabel('mfft')
        plt.tight_layout()
        plt.savefig(results_folder+'stats_per_segm_region.png')
        plt.close()


##--------------------SAVING QUANTITATIVE METRICS DATA------------------------
print('Saving quantitative metrics in file...')
zcr_list = {}
zcr_list['NAA'] = {}
zcr_list['NAA']['mean'] = list(zcr_['NAA']['mean'])
zcr_list['NAA']['std'] = list(zcr_['NAA']['std'])
zcr_list['GABA'] = {}
zcr_list['GABA']['mean'] = list(zcr_['GABA']['mean'])
zcr_list['GABA']['std'] = list(zcr_['GABA']['std'])
zcr_list['Glx'] = {}
zcr_list['Glx']['mean'] = list(zcr_['Glx']['mean'])
zcr_list['Glx']['std'] = list(zcr_['Glx']['std'])

if save_pictures_along_the_way == True or save_pictures_stats == True:
    qnttive_file = results_folder+name_of_study+'.txt'
else:
    qnttive_file = name_of_study+'.txt'
with open(qnttive_file, "w") as f:
    f.write("fwhm_mfft=")
    f.write(json.dumps(fwhm_mfft))
    f.write('\n')
    f.write("fwhm_mfft_real=")
    f.write(json.dumps(fwhm_mfft_real))
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
        f.write("qntty_absolute_regions=")
        f.write(json.dumps(qntty_absolute_regions))
        f.write('\n')
        f.write("stats_per_region=")
        f.write(json.dumps(stats_per_region))
        f.write('\n')
        f.write("stats_global=")
        f.write(json.dumps(stats_global))

print('All done!')