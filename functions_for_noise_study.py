import numpy as np
from utils import give_idx_ppm_point_for_different_ppm_arrays, give_idx_time_point_for_different_time_arrays, get_metrics


def spect_noise_estimation(list_spects,list_ppm_arrays,list_ppm_regions,part,degree):
  """
  Get average polynomial fit for a list of spectrum given a list of ppm value.
  Inputs:
  list_spects: list of (N,f) arrays, every line in the array list_spects[i][line_j,:] contains a spectrum or something similar, i.e., a function of ppm
  list_ppm_arrays: list of arrays, list_ppm_arrays[i] corresponds to the ppm axis for the multi-dim array in list_spects[i]
  list_ppm_regions: list of lists, every list within list_ppm_regions should have 2 values, the starting ppm point and the ending ppm point of that region (allows estimation of noise level in multiple regions)
  part: str, if 'imag' -> consider imaginary part of image, 'abs' -> consider absolute part of image
                'phase' -> consider phase of image, else consider real part
  degree: int, degree of polynome to fit
  Output:
  std_measures: dict of lists, std_measures['ppm1:ppm2'] gives a list with the average std level measured to each spect in list_spects at the region defined by ppm points : ppm1 and ppm2 (one of the lists in list_ppm_regions),
                also gives the std of the mean, and average coefficients for the poly fit for each region
  """
  std_measures = {}
  for reg in list_ppm_regions:

    if reg[0] <= reg[1]:
      smaller_value = reg[0]
      higher_value = reg[1]
    elif reg[0] > reg[1]:
      smaller_value = reg[1]
      higher_value = reg[0]

  
    idx_start = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm_arrays,ppm_point=smaller_value)
    idx_end = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm_arrays,ppm_point=higher_value)
    std_measures[str(smaller_value)+':'+str(higher_value)] = {'mean':[],'std':[], 'avg_coefs':[]}

    for i in range(len(list_spects)):
      if part == 'abs':
        obj = np.abs(list_spects[i][:,idx_start[i]:idx_end[i]])
      elif part == 'phase':
        obj = np.angle(list_spects[i][:,idx_start[i]:idx_end[i]],False)
      elif part == 'imag':
        obj = np.imag(list_spects[i][:,idx_start[i]:idx_end[i]])
      else:
        obj = np.real(list_spects[i][:,idx_start[i]:idx_end[i]])
      
      aux_ppm = np.flip(list_ppm_arrays[i])[idx_start[i]:idx_end[i]]
      std_estimated_aux = []
      coefs_est = np.empty((obj.shape[0],degree+1))
      for k in range(obj.shape[0]):
        coef = np.polynomial.polynomial.polyfit(x=aux_ppm,y=obj[k,:],deg=degree)
        estimate = np.zeros(aux_ppm.shape)
        for d in range(degree+1):
          estimate = estimate + coef[d]*(aux_ppm**d)
        detrending = obj[k,:]-estimate
        std_estimated_aux.append(np.std(detrending))
        coefs_est[k,:] = coef
      std_measures[str(smaller_value)+':'+str(higher_value)]['mean'].append(np.mean(np.array(std_estimated_aux)))
      std_measures[str(smaller_value)+':'+str(higher_value)]['std'].append(np.std(np.array(std_estimated_aux)))
      std_measures[str(smaller_value)+':'+str(higher_value)]['avg_coefs'].append(np.mean(coefs_est,axis=0))

  return std_measures


def stats_per_masked_regions(masks,spgram, part):
    """
    Get stats of masked regions in spgram.
    Inputs:
    masks: list of arrays, masks[i] should contain a binary array of size (N,f,t)
    spgram: array of size (N,f,t)
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    Outputs:
    stats_per_region: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                  for every key there is another dict:
                    stats_per_region['mean'] get keys ['0','1',..,'i',...], where 'i' is the mask index
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
    
    for mask_idx, mask in enumerate(masks):
        aux_list = []
        for j in range(spgram.shape[0]):
            masked_obj = (obj[j,:,:]*mask[j,:,:]).ravel()
            if np.all(masked_obj == 0) == False:
                masked_obj = masked_obj[masked_obj != 0]
                aux_list.append(masked_obj)                
            else:
                print('a region without any segmented object has been found, you might want to rethink your mask if this is not expected')
        dict_metrics_aux = get_metrics(list_of_interest=aux_list)    

        stats_per_region['mean'][str(mask_idx)] = dict_metrics_aux['mean']
        stats_per_region['median'][str(mask_idx)] = dict_metrics_aux['median']
        stats_per_region['std'][str(mask_idx)] = dict_metrics_aux['std']
        stats_per_region['skewness'][str(mask_idx)] = dict_metrics_aux['skewness']
        stats_per_region['kurtosis'][str(mask_idx)] = dict_metrics_aux['kurtosis']

    return stats_per_region

def stats_per_masked_regions_for_different_spgrams(masks,spgram_dict, part):
    """
    Get the stats for the masked regions in spectrograms generated by varying parameters.
    Inputs:
    masks: list of arrays, masks[i] should contain a binary array of size (N,f,t)
    spgram_dict: dict of lists, dict with keys 'param_x'.
                 For each key there is a list of 4 objects: [spectrogram of size (N,f,t), frequency array of size (f,), ppm array of size (f,), time array of size (t,)]
    part: if 'imag' -> consider imaginary part of spgram, 'abs' -> consider absolute part of spgram
                'phase' -> consider phase of spgram, else consider real part
    Outputs:
    stats_per_region: dict with keys 'mean','median','std','skewness' and 'kurtosis'.
                  for every key there is another dict with keys with the mask index
                  stats_per_region['mean'] get keys ['0','1',..,'i',...], where 'i' is the mask index
                      for every region key there is another inner dict with keys 'mean' and 'std' for the mean and std of 
                      the metric being considered
    """
    stats_per_region = {'mean':{},'median':{},'std':{},'skewness':{},'kurtosis':{}}
    for j in range(len(list(stats_per_region.keys()))):
        stats_per_region[list(stats_per_region.keys())[j]] = {}
        for k in range(len(masks)):
          stats_per_region[list(stats_per_region.keys())[j]][str(k)] = {'mean':[],'std':[]}
            
    for i in range(len(list(spgram_dict.keys()))):
        stats_per_region_aux = stats_per_masked_regions(masks=masks,spgram=spgram_dict[list(spgram_dict.keys())[i]][0],part=part)

        for k in range(len(masks)):
          for j in range(len(list(stats_per_region.keys()))):
            stats_per_region[list(stats_per_region.keys())[j]][str(k)]['mean'].append(stats_per_region_aux[list(stats_per_region.keys())[j]][str(k)]['mean'])
            stats_per_region[list(stats_per_region.keys())[j]][str(k)]['std'].append(stats_per_region_aux[list(stats_per_region.keys())[j]][str(k)]['std'])
    
    return stats_per_region

