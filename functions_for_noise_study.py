import numpy as np
from utils import give_idx_ppm_point_for_different_ppm_arrays

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
  std_measures: dict of lists, std_measures['ppm1:ppm2'] gives a list with the average std level measured to each spect in list_spects at the region defined by ppm points : ppm1 and ppm2 (one of the lists in list_ppm_regions)
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

