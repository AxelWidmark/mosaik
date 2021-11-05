import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
pi=math.pi
import os
import tensorflow as tf
numTFThreads = 0
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.compat.v1.disable_eager_execution()


# bounds for the population parameter prior
rho_bounds = [0., 0.3]
zsun_bounds = [-0.05, 0.05]
wsun_bounds = [-20., 20.]
mean_bounds = [-250., 250]
disp_bounds = [0., 250.]
accum_w_disp_bounds = [0., 50.]

# some functions in non-TF language (mostly not used, as they are also defined as Tf-funcs)
def sigmoid(x, bounds=[0, 1]):
    return bounds[0] + (bounds[1] - bounds[0]) * 1./(1.+np.exp(-x))
def sigmoid_inv(xx, bounds=[0, 1]):
    x = (xx-bounds[0]) / (bounds[1]-bounds[0])
    return -np.log(1./x-1.)
def one_way(zs):  
    fac = np.concatenate((1 - zs, np.array([1])))
    zsb = np.concatenate((np.array([1]), zs))
    fs = np.cumprod(zsb) * fac
    return fs
def other_way(xx):
    def fun(z):
        return np.sum((one_way(z) - xx)**2.)
    res = minimize(fun, x0=np.repeat(0.5, xx.size-1))
    return res.x
def get_pop_params(vector, num_vel_gaussians=10):
        rho_params = sigmoid(vector[0:4], rho_bounds)
        z_sun = sigmoid(vector[4], zsun_bounds)
        w_sun = sigmoid(vector[5], wsun_bounds)
        vel_amplitudes = one_way( sigmoid(vector[6:5+num_vel_gaussians]) )
        vel_disps_u = sigmoid(vector[5+1*num_vel_gaussians:5+2*num_vel_gaussians], disp_bounds)
        vel_disps_v = sigmoid(vector[5+2*num_vel_gaussians:5+3*num_vel_gaussians], disp_bounds)
        vel_disps_w = np.cumsum( sigmoid(vector[5+3*num_vel_gaussians:5+4*num_vel_gaussians], accum_w_disp_bounds) )
        vel_disps = np.transpose( [vel_disps_u, vel_disps_v, vel_disps_w] )
        vel_means_u = sigmoid(vector[5+4*num_vel_gaussians:5+5*num_vel_gaussians], mean_bounds)
        vel_means_v = sigmoid(vector[5+5*num_vel_gaussians:5+6*num_vel_gaussians], mean_bounds)
        vel_means = np.transpose( [vel_means_u, vel_means_v] )
        return rho_params, z_sun, w_sun, vel_amplitudes, vel_means, vel_disps




class tffunc():
    def __init__(self, df, eff_area, eff_area_z_vec, num_vel_gaussians=10, rho_scales=[0.040*2**i for i in range(4)]):
        self.num_vg = num_vel_gaussians
        self.rho_scales = rho_scales
        self.df = df
        self.num_stars = self.df.shape[0]
        self.Num_stars = tf.constant(self.num_stars, dtype=tf.float64)
        # STELLAR CONSTANTS (D for DATA)
        self.D_appG = tf.constant(self.df['phot_g_mean_mag'], dtype=tf.float64)
        self.D_l = tf.constant(pi/180.*self.df['l'], dtype=tf.float64)
        self.D_b = tf.constant(pi/180.*self.df['b'], dtype=tf.float64)
        self.D_par = tf.constant(self.df['parallax'], dtype=tf.float64)
        self.D_pmra = tf.constant(self.df['pmra'], dtype=tf.float64)
        self.D_pmde = tf.constant(self.df['pmdec'], dtype=tf.float64)
        self.D_vRV = tf.constant(np.nan_to_num(self.df['radial_velocity'], nan=0.), dtype=tf.float64)
        self.D_parerr = tf.constant(self.df['parallax_error'], dtype=tf.float64)
        self.D_pmraerr = tf.constant(self.df['pmra_error'], dtype=tf.float64)
        self.D_pmdeerr = tf.constant(self.df['pmdec_error'], dtype=tf.float64)
        self.D_transf_pm_coeffA = tf.constant(df['transf_pm_coeffA'], dtype=tf.float64)
        self.D_transf_pm_coeffB = tf.constant(df['transf_pm_coeffB'], dtype=tf.float64)
        self.D_vRVerr = tf.constant(np.nan_to_num(self.df['radial_velocity_error'], nan=20.), dtype=tf.float64)
        self.D_vRV_incl_indices = tf.constant(np.where(self.df['radial_velocity'].notna().values)[0], dtype=tf.int32)
        self.D_vRV_excl_indices = tf.constant(np.where(self.df['radial_velocity'].isnull().values)[0], dtype=tf.int32)
        small_errors = np.where((self.df['radial_velocity_error']<2.) & (self.df['parallax_error']<.2))[0]
        small_errors_inv = np.where(np.invert((self.df['radial_velocity_error']<2.) & (self.df['parallax_error']<.2)))[0]
        self.D_small_errs_incl_indices = tf.constant(small_errors, dtype=tf.int32)
        self.D_small_errs_excl_indices = tf.constant(small_errors_inv, dtype=tf.int32)
        print('subset with small errors:', len(small_errors), 'of total', self.num_stars)
        print('subset with missing RVs:', len(np.where(self.df['radial_velocity'].isnull().values)[0]), 'of total', self.num_stars)
        mulmubparcorr_Ms = np.array( [ [                                    \
            [1., self.df['parallax_pmra_corr'][i], self.df['parallax_pmdec_corr'][i]],           \
            [self.df['parallax_pmra_corr'][i], 1., self.df['pmra_pmdec_corr'][i]],   \
            [self.df['parallax_pmdec_corr'][i], self.df['pmra_pmdec_corr'][i], 1.]    \
            ] for i in df.index] )
        mulmubparcorr_invMs = np.array( [np.linalg.inv(m) for m in mulmubparcorr_Ms] )
        self.D_invcorrM = tf.constant(mulmubparcorr_invMs, dtype=tf.float64)
        # NOTE: rot_M * uvw = vlvbvr
        rot_matrices = np.array( [ [                                    \
            [-np.sin(l),           +np.cos(l),           +0.],           \
            [-np.sin(b)*np.cos(l), -np.sin(b)*np.sin(l), +np.cos(b)],   \
            [+np.cos(b)*np.cos(l), +np.cos(b)*np.sin(l), +np.sin(b)]    \
            ] for l,b in pi/180.*df[['l','b']].values ] )
        rot_matrices_inv = np.array( [np.linalg.inv(m) for m in rot_matrices] )
        self.D_rotM = tf.constant(rot_matrices, dtype=tf.float64)
        self.D_rotM_inv = tf.constant(rot_matrices_inv, dtype=tf.float64)
        self.Rho_scales = tf.constant(rho_scales, dtype=tf.float64)
        # NORMALISATION
        self.eff_area = eff_area
        self.eff_area_z_vec = eff_area_z_vec
        self.Eff_area = tf.constant(self.eff_area, dtype=tf.float64)
        self.Eff_area_z_vec = tf.constant(self.eff_area_z_vec, dtype=tf.float64)
        # Eff_area_w_vec is used for normalisation when mirror symm. is broken
        self.Eff_area_w_vec = tf.constant(np.linspace(0., 300., 301), dtype=tf.float64) # np.linspace(0.05, 299.95, 3000)
        self.num_pop_params = len(self.rho_scales) + 1 + 6*self.num_vg
        self.num_stellar_params = 4*self.num_stars
        self.num_params = self.num_pop_params + self.num_stellar_params
        self.var_bounds = np.concatenate([ [rho_bounds for i in range(len(self.rho_scales))]                                    , \
                                      [zsun_bounds]                                                     , \
                                      [wsun_bounds]                                                     , \
                                      [[0., 1.] for i in range(self.num_vg-1)]                          , \
                                      [disp_bounds for i in range(2*self.num_vg)]                       , \
                                      [accum_w_disp_bounds for i in range(self.num_vg)]               , \
                                      [mean_bounds for i in range(2*self.num_vg)], \
                                    ])
        self.Var_bounds = tf.constant(self.var_bounds, dtype=tf.float64)
    
    # 
    @tf.function
    def Vector_2_pop_params(self, Vector):
        Rho_params = rho_bounds[0] + (rho_bounds[1]-rho_bounds[0]) * tf.sigmoid(Vector[0:len(self.rho_scales)])
        Z_sun = zsun_bounds[0] + (zsun_bounds[1]-zsun_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)])
        W_sun = wsun_bounds[0] + (wsun_bounds[1]-wsun_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)+1])
        Vel_amplitudes = self.Add_to_1( tf.sigmoid(Vector[len(self.rho_scales)+2:len(self.rho_scales)+1+self.num_vg]) )
        Vel_disps_u = disp_bounds[0] + (disp_bounds[1]-disp_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)+1+1*self.num_vg:len(self.rho_scales)+1+2*self.num_vg])
        Vel_disps_v = disp_bounds[0] + (disp_bounds[1]-disp_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)+1+2*self.num_vg:len(self.rho_scales)+1+3*self.num_vg])
        Vel_disps_w = tf.math.cumsum(accum_w_disp_bounds[0] + (accum_w_disp_bounds[1]-accum_w_disp_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)+1+3*self.num_vg:len(self.rho_scales)+1+4*self.num_vg]))
        Vel_disps = tf.transpose( tf.stack([Vel_disps_u, Vel_disps_v, Vel_disps_w]) )
        Vel_means = tf.reshape( mean_bounds[0] + (mean_bounds[1]-mean_bounds[0]) * tf.sigmoid(Vector[len(self.rho_scales)+1+4*self.num_vg:len(self.rho_scales)+1+6*self.num_vg]), \
                    [self.num_vg, 2] )
        return Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps
        
    
    # GRAVITATIONAL POTENTIAL
    @tf.function
    def Phi_of_z(self, z, Rho_params):
        return 2e6/37.*tf.reduce_sum( Rho_params[:,None]*tf.math.log(tf.cosh(z[None,:]/self.Rho_scales[:,None]))*tf.pow(self.Rho_scales[:,None], 2), axis=0)
    
    
    # make set of params add up to one
    @tf.function
    def Add_to_1(self, zs):
        fac = tf.concat([1-zs, tf.constant([1], tf.float64)], 0)
        zsb = tf.concat([tf.constant([1], tf.float64), zs], 0)
        return tf.math.cumprod(zsb) * fac
    
    
    def get_star_coords(self, vector):
        Vector = tf.Variable(vector, dtype=tf.float64)
        X_par, X_pmra, X_pmde, X_vRV, S_par, S_absG, S_xyz, S_uvw = self.Vector_2_star_params(Vector)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        #sess = tf.compat.v1.Session(config=session_conf)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            s_xyz, s_uvw, s_absG = sess.run( [S_xyz, S_uvw, S_absG] )
        df_stars = pd.DataFrame({'x':s_xyz[:,0], 'y':s_xyz[:,1], 'z':s_xyz[:,2], 'u':s_uvw[:,0], 'v':s_uvw[:,1], 'w':s_uvw[:,2], 'absG':s_absG})
        return df_stars
    
    
    # gives stellar parameters
    @tf.function
    def Vector_2_star_params(self, Vector):
        Stars_vector = Vector[self.num_pop_params: self.num_pop_params + 4*self.num_stars]
        # these are defined as outliers w.r.t. the observed values and uncertainties
        X_par =  Stars_vector[0*self.num_stars: 1*self.num_stars]
        X_pmra = Stars_vector[1*self.num_stars: 2*self.num_stars]
        X_pmde = Stars_vector[2*self.num_stars: 3*self.num_stars]
        X_vRV =  Stars_vector[3*self.num_stars: 4*self.num_stars]
        S_par = self.D_par + X_par*self.D_parerr
        S_dist = 1./S_par
        S_absG = self.D_appG - 5.*tf.math.log(S_dist/0.01)/tf.math.log(tf.constant(10., dtype=tf.float64))
        S_pmra = self.D_pmra + X_pmra*self.D_pmraerr
        S_pmde = self.D_pmde + X_pmde*self.D_pmdeerr
        S_mul = self.D_transf_pm_coeffA*S_pmra + self.D_transf_pm_coeffB*S_pmde
        S_mub = -self.D_transf_pm_coeffB*S_pmra + self.D_transf_pm_coeffA*S_pmde
        S_vRV = self.D_vRV + X_vRV*self.D_vRVerr
        S_xyz = tf.transpose( tf.stack([ S_dist*tf.cos(self.D_l)*tf.cos(self.D_b), S_dist*tf.sin(self.D_l)*tf.cos(self.D_b), S_dist*tf.sin(self.D_b) ]) )
        S_vlvbvr = tf.transpose( tf.stack([ 4.74057*S_dist*S_mul, 4.74057*S_dist*S_mub, S_vRV ]) )
        S_uvw = tf.linalg.matvec(self.D_rotM_inv, S_vlvbvr)
        return X_par, X_pmra, X_pmde, X_vRV, S_par, S_absG, S_xyz, S_uvw
    
    
    # calculates normalisation
    @tf.function
    def Normalisation(self, Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps):
        Phi_vec = self.Phi_of_z(self.Eff_area_z_vec+Z_sun[None], Rho_params)
        Stellar_number_density_vec = tf.reduce_sum( Vel_amplitudes[:,None] * \
            tf.exp( -Phi_vec[None,:]/tf.pow(Vel_disps[:,2,None], 2) ), [0] )
        Log_norm = tf.math.log( tf.reduce_sum(Stellar_number_density_vec*self.Eff_area) )
        return Log_norm
    
    
    
    
    @tf.function
    def Log_posterior(self, Vector, neglect_vels=False):
        # POPULATION PARAMETERS
        Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps = self.Vector_2_pop_params(Vector[0:self.num_pop_params])
        if neglect_vels:
            Vels_included = self.D_small_errs_incl_indices
            Vels_excluded = self.D_small_errs_excl_indices
        else:
            Vels_included = tf.constant(np.arange(self.num_stars), dtype=tf.int32)
            Vels_excluded = tf.constant([], dtype=tf.int32)
        
        X_par, X_pmra, X_pmde, X_vRV, S_par, S_absG, S_xyz, S_uvw = self.Vector_2_star_params(Vector)
        S_z = S_xyz[:,2]

        ### COMPUTE PROBS ###
        # NORMALISATION
        Log_norm = self.Normalisation(Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps)
        
        # DATA LIKELIHOOD
        if neglect_vels:
            Log_pr_vRV = tf.reduce_sum( - 0.5 * ( tf.gather(tf.pow(X_vRV, 2), Vels_included) ), [0] )
        else:
            Log_pr_vRV = tf.reduce_sum( - 0.5 * ( tf.gather(tf.pow(X_vRV, 2), self.D_vRV_incl_indices) ), [0] )
        Log_pr_parpmpm = tf.reduce_sum( tf.gather( - 0.5 * (                         \
                        self.D_invcorrM[:,0,0]*tf.pow(X_par, 2) + self.D_invcorrM[:,1,1]*tf.pow(X_pmra, 2) + self.D_invcorrM[:,2,2]*tf.pow(X_pmde, 2) +      \
                        + 2.*self.D_invcorrM[:,0,1]*X_par*X_pmra + 2.*self.D_invcorrM[:,0,2]*X_par*X_pmde + 2.*self.D_invcorrM[:,1,2]*X_pmra*X_pmde            \
                        ), Vels_included), [0] )
        Log_likelihood = Log_pr_parpmpm + Log_pr_vRV
        Log_jacobian = tf.reduce_sum( tf.gather( - 6. * tf.math.log(S_par), Vels_included) ) +   \
                       tf.reduce_sum( tf.gather( - 4. * tf.math.log(S_par), Vels_excluded) )


        # PROBABILITY OF BRIGHTNESS
        # this prevents objects from being very far away (and thus extremely bright, in abs terms)
        Log_brightness = tf.reduce_sum( tf.math.log( tf.tanh((S_absG-3.7012402)/1.74501757) + 1. ) )
        
        # PROBABILITY OF PHASE SPACE COORDS
        S_midW = tf.sqrt( tf.pow(S_uvw[:,2]+W_sun[None], 2) + 2.*self.Phi_of_z(S_z+Z_sun[None], Rho_params) )
        Log_pr_uvw = tf.reduce_sum( tf.gather( tf.reduce_logsumexp(     \
                         tf.math.log(Vel_amplitudes[:,None]) - tf.math.log(Vel_disps[:,0,None]*Vel_disps[:,1,None]*Vel_disps[:,2,None])      \
                         - 0.5 * ( tf.pow(S_uvw[None,:,0]-Vel_means[:,0,None], 2)/tf.pow(Vel_disps[:,0,None], 2) +     \
                                   tf.pow(S_uvw[None,:,1]-Vel_means[:,1,None], 2)/tf.pow(Vel_disps[:,1,None], 2) +     \
                                   tf.pow(S_midW[None,:], 2)/tf.pow(Vel_disps[:,2,None], 2)   )                        \
                     , [0]), Vels_included) )
        if neglect_vels:
            Log_pr_z = tf.reduce_sum( tf.math.log( tf.reduce_sum( Vel_amplitudes[:,None] * \
                       tf.exp( -self.Phi_of_z(Z_sun[None]+tf.gather(S_z, Vels_excluded), Rho_params)[None,:]/tf.pow(Vel_disps[:,2,None], 2) ), [0] ) ), [0])
            Log_pr_phase_space = Log_pr_uvw + Log_pr_z
        else:
            Log_pr_phase_space = Log_pr_uvw
        
        # PRIOR
        Log_prior = tf.reduce_sum( -tf.pow(2./Vel_disps, 4) )
        LogCoordTransPrior = tf.reduce_sum( tf.math.log( (self.Var_bounds[:,1]-self.Var_bounds[:,0]) * \
                tf.exp(-Vector[0:self.num_pop_params]) / tf.pow(tf.exp(-Vector[0:self.num_pop_params])+1., 2) ) )
        
        Log_posterior_sum = Log_prior + LogCoordTransPrior + Log_pr_phase_space + Log_brightness + Log_jacobian + Log_likelihood - self.num_stars*Log_norm
        return Log_posterior_sum
    
    
    def get_log_posterior_value(self, vector, neglect_vels=False):
        Vector = tf.Variable(vector, dtype=tf.float64)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        #sess = tf.compat.v1.Session(config=session_conf)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            res = sess.run( self.Log_posterior(Vector, neglect_vels=neglect_vels) )
        return res
    
    
    def get_vector_guess(self):
        pop_params = np.concatenate( (np.random.normal(loc=-2.5, scale=0.5, size=len(self.rho_scales)), np.random.normal(loc=0., scale=0.1, size=1+self.num_vg), np.random.normal(loc=0., scale=0.5, size=5*self.num_vg)) )
        stellar_params = 0. * np.random.normal(size=self.num_stellar_params)
        vector = np.concatenate((pop_params, stellar_params))
        return vector
    
    
    def get_stepsize_guess(self):
        return np.concatenate([1e-4*np.ones(self.num_pop_params), 1e-1*np.ones(self.num_stellar_params)])
    
    
    # minimize posterior function
    def minimize_posterior(self, p0=None, fixed_stellar_params=False, neglect_vels=True, number_of_iterations=20000, print_gap=1000, numTFTthreads=0, learning_rate=1e-3):
        if p0 is None:
            vector = self.get_vector_guess()
        else:
            vector = p0
        if fixed_stellar_params:
            Vector_pop = tf.Variable(vector[0:self.num_pop_params], dtype=tf.float64)
            Stellar_consts = tf.constant(np.zeros(self.num_stellar_params), dtype=tf.float64)
            Vector = tf.concat([Vector_pop, Stellar_consts], 0)
        else:
            Vector = tf.Variable(vector, dtype=tf.float64)
        Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps = self.Vector_2_pop_params(Vector)
        MinusLogPosterior = -self.Log_posterior(Vector, neglect_vels=neglect_vels)
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(MinusLogPosterior)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            for i in range(number_of_iterations):
                _, minusLogPosterior, vector, rho_params, z_sun, w_sun, vel_amplitudes, vel_means, vel_disps  =  \
                    sess.run([Optimizer, MinusLogPosterior, Vector, Rho_params, Z_sun, W_sun, Vel_amplitudes, Vel_means, Vel_disps])
                if np.isnan(minusLogPosterior):
                    print(minusLogPosterior)
                    print(list(vector[0:self.num_pop_params]), "\n\n")
                    raise ValueError("AdamOptimizer returned NaN")
                if i%print_gap==0:
                    print(minusLogPosterior)
                    print(np.sum(rho_params))
                    print(list(rho_params), z_sun, w_sun, "\n\n")
        return minusLogPosterior, vector
    
    
    # run MCMC chain
    def run_HMC(self, p0=None, steps=1e4, burnin_steps=0, num_adaptation_steps=0, num_leapfrog_steps=3, \
                step_size_start=None, num_steps_between_results=0): #, noUturn=False
        #tf.compat.v1.disable_eager_execution()
        if p0 is None:
            vector = self.get_vector_guess()
        else:
            vector = p0
        if step_size_start is None:
            step_size_start = self.get_stepsize_guess()
        the_function = self.Log_posterior
        num_results = int(steps)
        num_burnin_steps = int(burnin_steps)
        adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=the_function,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size_start),
            num_adaptation_steps=int(num_adaptation_steps))
        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, [is_accepted, step_size, log_prob] = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=np.array(vector, dtype=np.float64),
                kernel=adaptive_hmc,
                num_steps_between_results=num_steps_between_results,
                trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted, pkr.inner_results.accepted_results.step_size, pkr.inner_results.accepted_results.target_log_prob])
            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float64))
            return samples, is_accepted, step_size, log_prob
        Samples, Is_accepted, Step_size, Log_prob = run_chain()
        with tf.compat.v1.Session() as sess:
            samples, is_accepted, step_size, log_prob = sess.run([Samples, Is_accepted, Step_size, Log_prob])
        return samples, log_prob, step_size, is_accepted
    
    
    # takes a chain and returns a stepsize vector
    def adjust_stepsize(self, samples):
        res = 1e-1 * np.std(samples, axis=0)
        if np.sum(res)==0.:
            res = self.get_stepsize_guess()
        return res
