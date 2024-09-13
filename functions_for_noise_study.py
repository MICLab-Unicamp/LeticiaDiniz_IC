import numpy as np
from functions_for_param_study import give_idx_ppm_point_for_different_ppm_arrays

"""
def spect_noise_estimation(list_spects,list_ppm_arrays,list_ppm_regions,part):

  Get average polynomial fit (pol degree = 2) for a list of spectrum given a list of ppm value.
  Inputs:
  list_spects: list of (N,f) arrays, every line in the array list_spects[i][line_j,:] contains a spectrum or something similar, i.e., a function of ppm
  list_ppm_arrays: list of arrays, list_ppm_arrays[i] corresponds to the ppm axis for the multi-dim array in list_spects[i]
  list_ppm_regions: list of lists, every list within list_ppm_regions should have 2 values, the starting ppm point and the ending ppm point of that region (allows estimation of noise level in multiple regions)
  part: str, if 'imag' -> consider imaginary part of image, 'abs' -> consider absolute part of image
                'phase' -> consider phase of image, else consider real part
  Output:
  std_measures: list of lists, std_measures[k] gives a list with the average std level measured to each spect in list_spects at the region list_ppm_regions[k]

  for reg in list_ppm_regions:
    idx_start = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm_arrays,ppm_point=reg[0])
    idx_end = give_idx_ppm_point_for_different_ppm_arrays(list_ppm_arrays=list_ppm_arrays,ppm_point=reg[-1])
    for i in range(len(list_spects)):
      if part == 'abs':
        obj = np.abs(list_spects[i])
      elif part == 'phase':
        obj = np.angle(list_spects[i],False)
      elif part == 'imag':
        obj = np.imag(list_spects[i])
      else:
        obj = np.real(list_spects[i])

      spect_array = np.real(spect[i,idx_noise_2:idx_noise_1])



    idx_noise_1 = np.abs(ppm[i,:] - 8.5).argmin()
    idx_noise_2 = np.abs(ppm[i,:] - 9.5).argmin()
    idx_noise_3 = np.abs(ppm[i,:] - 10.5).argmin()

    #assumes ppm is inverted: smaller values in higher indexes
    ppm_array_1 = ppm[i,idx_noise_2:idx_noise_1]
    ppm_array_2 = ppm[i,idx_noise_3:idx_noise_2]
    spect_array_1 = np.real(spect[i,idx_noise_2:idx_noise_1])
    spect_array_2 = np.real(spect[i,idx_noise_3:idx_noise_2])

    estimate_1 = np.polyfit(ppm_array_1, spect_array_1, 2)
    estimate_2 = np.polyfit(ppm_array_2, spect_array_2, 2)
    aux_1 = (estimate_1[0]*(ppm_array_1**2)) + (estimate_1[1]*ppm_array_1) +  estimate_1[2]
    aux_2 = (estimate_2[0]*(ppm_array_2**2)) + (estimate_2[1]*ppm_array_2) +  estimate_2[2]
    detrending_1 = spect_array_1 - aux_1
    detrending_2 = spect_array_2 - aux_2
    std_1 = np.std(detrending_1)
    std_2 = np.std(detrending_2)

    if np.abs(std_1) < np.abs(std_2):
      std = std_1
    else:
      std = std_2

    std_array[i] = std

  return std_array
"""
