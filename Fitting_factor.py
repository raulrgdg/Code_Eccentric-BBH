'''
This code has been developed by Raúl Rodríguez for his Final Degree Project at USC, under the supervision of Thomas Dent.
The final purpose of this code is to find the fitting factor for a population of eccentric signals.
'''

import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc.filter import matchedfilter
from scipy.optimize import differential_evolution
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
import time
start_time = time.time()

#Generating template (non-eccentric waveform)
def template_waveform(mass1,mass2):
    hp_non_eccen, _ = get_fd_waveform(approximant="TaylorF2",
                                       mass1=mass1,
                                       mass2=mass2,
                                       f_lower=20.0,
                                       delta_f=1.0)
    hp_non_eccen.resize(1024) #Same lenght in template and signal 
    hp_non_eccen /= max(abs(hp_non_eccen)) 

    return hp_non_eccen

#Final function: obtaining the fitting factor and optimal template parameters for a given signal.
def maxoverlap_sig_tem(s_q, s_mchirp, e):
    #Signal
    s_m1 = mass1_from_mchirp_q(s_mchirp, s_q)
    s_m2 = mass2_from_mchirp_q(s_mchirp, s_q)
    s_e = e

    hp_eccen, _ = get_fd_waveform(approximant="TaylorF2Ecc",
                                    mass1=s_m1,
                                    mass2=s_m2,
                                    eccentricity=s_e,  
                                    spin1z=0.0,
                                    spin2z=0.0,
                                    f_lower=20.0,
                                    delta_f=1.0)
    hp_eccen.resize(1024)
    hp_eccen /= max(abs(hp_eccen))

    #Optimization method: Differential Evolution
    def objective_function(x):
        q,mchirp=x

        mass1 = mass1_from_mchirp_q(mchirp, q)
        mass2 = mass2_from_mchirp_q(mchirp, q)
    
        hp_non_eccen=template_waveform(mass1,mass2)   

        overlap, _ = match(hp_eccen, hp_non_eccen,  low_frequency_cutoff=20.0, high_frequency_cutoff=1024.0 ,psd=None, subsample_interpolation=True)
        return -overlap
    
    #Specifying the bounds for the mass ratio and the chirp mass of the template.
    bounds=[(0.1,1.0),(5.8,7.1)] 
    result_DE=differential_evolution(objective_function, bounds, popsize=35)
    if result_DE.success != True:
        print(f'The maximization could not be completed for the signal: s_e={s_e}, s_mchirp={s_mchirp} y s_q={s_q}')

    best_q=result_DE.x[0]
    best_mchirp=result_DE.x[1]
    best_match=-(result_DE.fun)

    return best_q, best_mchirp, best_match, hp_eccen

#Now, we establish the population of the signals whose fitting factor will be computed
num_elements = 2 #Length of the population
s_e=np.linspace(0.01, 0.6, num_elements) #Eccentricity range of the population
s_q=np.linspace(0.1,1.0, num_elements ) #Mass ratio range of the population

best_q = np.zeros((num_elements,num_elements))
best_mchirp = np.zeros((num_elements,num_elements))
best_overlap = np.zeros((num_elements,num_elements))

#Computing the fitting factor and saving the parameters of the best template for each signal
for i in range(num_elements):
    for j in range(num_elements):
        q,mchirp,overlap,t=maxoverlap_sig_tem(s_q[j],6.08364,s_e[i]) #Computed for a fixed chirp mass
        best_q[i,j] = q
        best_mchirp[i,j] = mchirp
        best_overlap[i,j] = overlap

end_time=time.time()

#As an example, we show here the running time for different population with my laptop
#num_elements=20; t= 2110.0369 s
#num_elements=30 ; t 4701.0021 s
#pnum_elements=45 ; t= 11441.0871 s

#Showing the results of the optimization
print('Results: \n')
for i in range(num_elements):
    for j in range(num_elements):
        print(f'For the signal: s_e={s_e[i]}, s_q={s_q[j]} and chirp mass=6.08364')
        print(f'The fitting factor is: {best_overlap[i,j]}')
        print(f'The best template has these parameters: q={best_q[i,j]}, chirp mass={best_mchirp[i,j]}','\n')

print(f"Running time: {end_time - start_time:.4f} s")

#Saving the matrix wiht the fitting factor (best_overlap) and the parameters of the template which best fit each signal
np.save('best_q.npy', best_q)
np.save('best_mchirp.npy', best_mchirp)
np.save('best_overlap.npy', best_overlap)



