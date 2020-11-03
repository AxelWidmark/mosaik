import math
import numpy as np
pi=math.pi
import sys



# CHOOSE YOUR STELLAR SAMPLE
sample_i = "A1"
# CHOOSE YOUR Z_LIM -- in the code, z_lim is given in kpc
z_lim = 0.2


### LOAD SAMPLE AND EFF. AREA
from LoadData import load_data, sample_name_2_cuts
R_cuts, l_cuts, absG_cuts = sample_name_2_cuts(sample_i)
sample = load_data(l_cuts, z_lim, R_cuts, absG_cuts)
sample.load_df()
df = sample.get_df()
eff_area, eff_area_z_vec = sample.get_eff_area()
sample_name = sample.sample_name




### IMPORT TENSORFLOW FUNCTIONS
from TfFunctions import tffunc
tff = tffunc(df, eff_area, eff_area_z_vec)
num_pop_params = tff.num_pop_params
# Initial minimization, 10 different starting points
minLogPost_MIN = np.inf
for i in range(10):
    minusLogPosterior, v_realisation = tff.minimize_posterior(fixed_stellar_params=True, neglect_vels=True, print_gap=1000, number_of_iterations=50000, learning_rate=1e-4)
    if not np.isnan(minusLogPosterior) and minusLogPosterior<minLogPost_MIN:
        minLogPost_MIN = minusLogPosterior
        vector = v_realisation
# HMC burn-in
step_size = tff.get_stepsize_guess()
# HMC burn-in
for ii in range(5):
    print("Burn-in round: "+str(ii+1))
    samples, log_prob, step_size, is_accepted = tff.run_HMC(p0=vector, steps=1e4, num_leapfrog_steps=3, step_size_start=step_size, burnin_steps=1e5, num_adaptation_steps=9e4)
    final_step_size = step_size[-1]
    step_size = tff.adjust_stepsize(samples)
    vector = samples[-1]
# Run chain
print("HMC, run chain")
samples, log_prob, step_size, is_accepted = tff.run_HMC(p0=vector, steps=1e4, num_steps_between_results=9, num_leapfrog_steps=3, step_size_start=step_size, burnin_steps=1e5, num_adaptation_steps=9e4)
final_step_size = step_size[-1]
step_size = tff.adjust_stepsize(samples)
vector = samples[-1]
# Save results
np.savez("../Results/chain_"+sample_name, pop_param_chain=np.array(samples, dtype=np.float32)[:,0:num_pop_params], final_step=np.array(samples, dtype=np.float32)[-1], log_prob=np.array(log_prob, dtype=np.float32), step_size=np.array(step_size, dtype=np.float32), final_step_size=np.array(final_step_size, dtype=np.float32)) 


print("\n\n\n", sample_name)
print("ALL DONE")