import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import healpy as hp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline
pi=math.pi
import os


# names of the stellar samples' area cells
allowed_names = ['A1', 'A2', 'A3', 'A4', \
                 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', \
                 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', \
                 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16']
# feed it an area cell name or integer and it returns the data cuts
def sample_name_2_cuts(name):
    if isinstance(name, int) and name>=0 and name<len(allowed_names):
        name = allowed_names[name]
    if name in allowed_names:
        tier = np.where(np.array( ['A','B','C','D'] ) == name[0])[0][0]
        anglenum = int(name[1:len(name)])-1
        l_cuts = [anglenum*360./(4*(tier+1)), (anglenum+1)*360./(4*(tier+1))]
        if tier==0:
            R_cuts = [0.05, 0.1]
            absG_cuts = [4.6, 5.5]
        elif tier==1:
            R_cuts = [0.1, 0.15]
            absG_cuts = [3.7, 5.0]
        elif tier==2:
            R_cuts = [0.15, 0.2]
            absG_cuts = [3.0, 4.7]
        elif tier==3:
            R_cuts = [0.2, 0.25]
            absG_cuts = [3.0, 4.5]
    else:
        raise Exception("The argument supplied to sample_name_2_cuts is not a valid sample name.")
    return R_cuts, l_cuts, absG_cuts




# this class makes it easy to handle the data of the stellar samples, and their effective areas
class load_data():
    
    def __init__(self, l_cuts, z_lim, R_cuts, absG_cuts):
        self.l_cuts = l_cuts
        self.z_lim = z_lim
        self.R_cuts = R_cuts
        self.absG_cuts = absG_cuts
        self.b_lim = np.arctan(self.z_lim/self.R_cuts[0])
        self.full_catalogue_path = "../../GAIA_DATA/Gaia_par-2_appG-8-14.csv"
        # z_high, R_low, R_high, absG_low, absG_high are the boundaries of the larger volume
        self.highest_parerr = 2.
        self.geom_max_dist = np.sqrt(self.z_lim**2+self.R_cuts[1]**2)
        self.geom_min_dist = self.R_cuts[0]
        self.z_high = np.min( [(1. - self.geom_max_dist*self.highest_parerr)**(-1) * self.z_lim, 1.] )
        self.R_high = 1. / (1./self.R_cuts[1] - self.highest_parerr)
        self.R_low = 1. / (1./self.R_cuts[0] + self.highest_parerr)
        self.max_absG_diff = 5.*np.log10(1.+self.highest_parerr*self.geom_max_dist)
        # absG_high and absG_low are the brightest and dimmest stars (in absolute terms)
        # that could be included given the maximum considered parallax error
        self.absG_high = self.absG_cuts[1] + self.max_absG_diff
        self.absG_low = self.absG_cuts[0] - self.max_absG_diff
        self.appG_high = self.absG_cuts[1] + 5.*np.log10(self.geom_max_dist/0.010)
        self.appG_low = self.absG_cuts[0] + 5.*np.log10(self.geom_min_dist/0.010)
    
    
    # returns the open cluster masks for a stellar sample
    def get_OC_masks(self):
        df_OCs = pd.read_csv('../Data/Cantat_Gaudin_clusters.csv')
        if np.mean(self.l_cuts)>270.:
            df_OCs['GLON'] = (df_OCs['GLON']-180.)%360. + 180.
        OC_angles_cut = (df_OCs['GLON']>self.l_cuts[0]-5.*df_OCs['r50']) & (df_OCs['GLON']<self.l_cuts[1]+5.*df_OCs['r50'])
        OC_distance_cut = (np.abs( np.sin(pi/180.*df_OCs['GLAT'])/df_OCs['plx'] ) < self.z_lim+0.05) & \
                          (np.cos(pi/180.*df_OCs['GLAT'])/df_OCs['plx'] > self.R_cuts[0]-0.05) & \
                          (np.cos(pi/180.*df_OCs['GLAT'])/df_OCs['plx'] < self.R_cuts[1]+0.05)
        OC_all_cuts = OC_angles_cut & OC_distance_cut
        OCmasks_centers = df_OCs[OC_all_cuts][['GLON', 'GLAT']].values # this is in unit degrees
        OCmasks_radii = 5.*df_OCs[OC_all_cuts]['r50'].values # this is in unit degrees
        l = df_OCs[OC_all_cuts]['GLON']
        b = df_OCs[OC_all_cuts]['GLAT']
        pmra = df_OCs[OC_all_cuts]['pmRA']
        pmdec = df_OCs[OC_all_cuts]['pmDE']
        print('l, b, pmra, pmdec:')
        print(l.values)
        print(b.values)
        print(pmra.values)
        print(pmdec.values)
        num_of_OCmasks = len(OCmasks_radii)
        return OCmasks_centers, OCmasks_radii, num_of_OCmasks
    
    
    def load_df(self, dust=True, maskOCs=True, save=True):
        self.maskOCs = maskOCs
        self.sample_name = str(self.l_cuts[0])+'-'+str(self.l_cuts[1])+'_'+  \
                      str(self.z_lim)+'_'+  \
                      str(self.R_cuts[0])+'-'+str(self.R_cuts[1])+'_'+  \
                      str(self.absG_cuts[0])+'-'+str(self.absG_cuts[1])+(not dust)*'_no-dust'+(self.maskOCs)*'_maskedOCs'
        print("Sample name:", self.sample_name)
        file_name = '../Data/sample_' + self.sample_name + '.csv'
        file_name_eff_area = '../Data/eff_area_sample_' + self.sample_name + '.npz'
        #print("File name:",file_name)
        if os.path.isfile(file_name and file_name_eff_area):
            self.df = pd.read_csv(file_name)
            npzfile = np.load(file_name_eff_area)
            self.eff_area = npzfile['eff_area']
            self.eff_area_z_vec = npzfile['eff_area_z_vec']
        else:
            df_full = pd.read_csv( self.full_catalogue_path ) # full catalogue here
            # Add parallax zero-point offset
            df_full['parallax'] += 0.03
            cutA = ( df_full['l']>self.l_cuts[0]) & (df_full['l']<self.l_cuts[1] ) & \
                ( np.abs( np.sin(pi/180.*df_full['b'])/df_full['parallax'] ) < self.z_lim ) & \
                ( np.cos(pi/180.*df_full['b'])/df_full['parallax'] > self.R_cuts[0] ) & \
                ( np.cos(pi/180.*df_full['b'])/df_full['parallax'] < self.R_cuts[1] )
            df = df_full[cutA][['ra', 'dec', 'l', 'b', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',    \
                'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'radial_velocity', 'radial_velocity_error', \
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']]
            df.reset_index()
            if self.maskOCs: # OPEN CLUSTER MASKS
                self.OCmasks_centers, self.OCmasks_radii, self.num_of_OCmasks = self.get_OC_masks()
                print('Number of OC masks:', self.num_of_OCmasks,'\n\n\n')
                print("OC centers:", self.OCmasks_centers)
                print("OC mask radii:", self.OCmasks_radii)                
                for i_OC in range(self.num_of_OCmasks):
                    i_OC_cut = np.array([180./pi*hp.rotator.angdist(ang_values, self.OCmasks_centers[i_OC], lonlat=True) \
                                for ang_values in df[['l', 'b']].values]) > self.OCmasks_radii[i_OC]
                    df = df[i_OC_cut]
                    df.reset_index()
            if dust:
                npzfile = np.load("../Data/3D_dust_interpolation.npz")
                x = 1e-3*npzfile['x']
                y = 1e-3*npzfile['y']
                z = 1e-3*npzfile['z']
                BmV_matrix = npzfile['BmV_matrix']
                # dust interpolation is only for heights |z| < 300 pc
                BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear', bounds_error=True) #, fill_value=None)
                xyz = np.transpose([ np.cos(pi/180.*df['l'])*np.cos(pi/180.*df['b'])/df['parallax'], \
                                     np.sin(pi/180.*df['l'])*np.cos(pi/180.*df['b'])/df['parallax'], \
                                     np.sin(pi/180.*df['b'])/df['parallax'] ])
                for xi in range(len(xyz)):
                    if np.abs(xyz[xi][2])>0.297:
                        xyz[xi] = xyz[xi]*0.297/np.abs(xyz[xi][2])
                df['dust_correction'] = 3.1*BmV_interp(xyz)
                cutB = (df['phot_g_mean_mag']-df['dust_correction'] - 5.*np.log10(100./df['parallax']) > self.absG_cuts[0]) &    \
                       (df['phot_g_mean_mag']-df['dust_correction'] - 5.*np.log10(100./df['parallax']) < self.absG_cuts[1])
            else:
                cutB = (df['phot_g_mean_mag'] - 5.*np.log10(100./df['parallax']) > self.absG_cuts[0]) &    \
                       (df['phot_g_mean_mag'] - 5.*np.log10(100./df['parallax']) < self.absG_cuts[1])
            df = df[cutB]
            df.reset_index()
            # mul = c1*pmra + c2*pmde
            # mub = -c2*pmra + c1*pmde 
            c1,c2 = self.transf_pm_coeffs(pi/180.*df['ra'], pi/180.*df['dec'])
            df['transf_pm_coeffA'] = c1
            df['transf_pm_coeffB'] = c2
            print("Sample shape:", df.shape)
            self.df = df
            if save:
                df.to_csv(file_name)
            # df_sub is a sample of stars in a data volume larger than the actual sample,
            # it is used in order to calculate the distribution of parallax uncertainties in the relevant volume
            self.df_sub = df_full[ (df_full['l']>self.l_cuts[0]) & (df_full['l']<self.l_cuts[1]) & \
                    (pi/180.*np.abs(df_full['b']) < self.b_lim) & \
                    (np.cos(pi/180.*df_full['b'])/df_full['parallax'] > self.R_cuts[0]) & \
                    (np.cos(pi/180.*df_full['b'])/df_full['parallax'] < self.R_cuts[1]) & \
                    (df_full['phot_g_mean_mag'] > self.appG_low) &    \
                    (df_full['phot_g_mean_mag'] < self.appG_high) ] \
                    [['l', 'b', 'parallax', 'parallax_error', 'phot_g_mean_mag']]
            print("Sub-sample shape:", self.df_sub.shape)
            print("\n\n\n\n\n\n")
            self.eff_area, self.eff_area_z_vec = self.compute_eff_area()
            if save:
                np.savez(file_name_eff_area, eff_area=self.eff_area, eff_area_z_vec=self.eff_area_z_vec)
        return 
    
    
    def get_df(self):
        return self.df
    
    def get_eff_area(self):
        return self.eff_area, self.eff_area_z_vec
    
    def transf_pm_coeffs(self, alpha, delta):
        # constants are from here: https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
        alphaNGP = 192.85948/180.*pi
        deltaNGP = 27.12825/180.*pi
        c1 = np.sin(deltaNGP)*np.cos(delta)-np.cos(deltaNGP)*np.sin(delta)*np.cos(alpha-alphaNGP)
        c2 = np.cos(deltaNGP)*np.sin(alpha-alphaNGP)
        norm = np.sqrt(c1**2.+c2**2.)
        c1 = c1/norm
        c2 = c2/norm
        return c1,c2
    
    
    def load_random_parerr_f(self):
        appGvec = np.linspace(self.appG_low, self.appG_high, int((self.appG_high-self.appG_low)/0.3+1.))
        appG_stepsize = (self.appG_high-self.appG_low) / float(len(appGvec)-1)
        bvec = np.linspace(-self.b_lim, self.b_lim, int(self.b_lim/0.2))
        b_stepsize = 2.*self.b_lim / float(len(bvec)-1)
        parerrpercentilesgrid = np.zeros((len(appGvec)-1, len(bvec)-1, 100))
        for i in range(len(appGvec)-1):
            cut = (self.df_sub['phot_g_mean_mag']>appGvec[i]) & (appGvec[i+1]>self.df_sub['phot_g_mean_mag'])
            for k in range(len(bvec)-1):
                cut = (self.df_sub['phot_g_mean_mag']>appGvec[i]) & \
                      (appGvec[i+1]>self.df_sub['phot_g_mean_mag']) & \
                      (pi/180.*self.df_sub['b'] >= bvec[k]) & \
                      (bvec[k+1] >= pi/180.*self.df_sub['b'])
                parerrpercentilesgrid[i][k] = np.percentile(self.df_sub[cut]['parallax_error'],np.linspace(0.5,99.5,100))
        def random_parerr_f(appG, b):
            if appG<self.appG_low or appG>self.appG_high:
                res = None
            else:
                appG_i = int( (appG-self.appG_low) / appG_stepsize)
                b_k = int( (b-bvec[0]) / b_stepsize)
                res = np.random.choice( parerrpercentilesgrid[appG_i][b_k] )
            return res
        return random_parerr_f
    
    
    def compute_eff_area(self, geometric=False, sample_size=1e7): # computing this with sample_size=1e7 should be around 10 hours
        tick0 = time.time()
        eff_area_z_vec = np.linspace(-self.z_high, self.z_high, int(2e3*self.z_high+1))
        if geometric:
            eff_area = np.ones(len(eff_area_z_vec))
        else:
            # completeness inferred from 2MASS xmatch, Gmag in 8--12 and 12--15
            npzfile = np.load("../Data/completeness.npz")
            map8to12 = npzfile['map8to12']
            map12to15 = npzfile['map12to15']
            def completeness(l,b,appG): # note that these angles are in radians!
                if appG>=8. and appG<12.:
                    res = hp.pixelfunc.get_interp_val(map8to12,b+pi/2.,l,nest=True)
                elif appG>=12. and appG<15.:
                    res = hp.pixelfunc.get_interp_val(map12to15,b+pi/2.,l,nest=True)
                else:
                    res = 0.
                return res
            
            # absG distribution
            npzfile = np.load('../Data/absG_histogram.npz')
            bin_means = npzfile['bin_means']
            histvals = npzfile['histvals']
            # PDF luminosity function in absG (not normalized)
            luminosityPDF = UnivariateSpline(bin_means,histvals,s=5e5)
            
            absG_vec = np.linspace(self.absG_low, self.absG_high, int((self.absG_high-self.absG_low)/0.1+1.))
            absG_accum = np.cumsum(luminosityPDF(absG_vec))
            absG_accum -= absG_accum[0]
            absG_accum /= absG_accum[-1]
            G_f = interp1d(absG_accum, absG_vec)
            def random_absG():
                return G_f( np.random.rand() )
            
            # parallax error distribution
            random_parerr_f = self.load_random_parerr_f()
            
            eff_area = np.zeros(len(eff_area_z_vec))
            z_bin_length = (np.max(eff_area_z_vec)-np.min(eff_area_z_vec)) / (len(eff_area_z_vec)-1)
            # rejection sampling stars here
            while np.sum(eff_area)<sample_size:
                R_rand = np.sqrt( np.random.uniform(self.R_low**2, self.R_high**2) )
                z_rand = np.random.uniform(-self.z_high, self.z_high)
                b_rand = np.arctan(z_rand/R_rand) # note that these angles are in radians!
                l_rand = pi/180.*np.random.uniform(self.l_cuts[0], self.l_cuts[1]) # note that these angles are in radians!
                if np.abs(b_rand)<self.b_lim:
                    dist_rand = np.sqrt( R_rand**2 + z_rand**2 )
                    absG_rand = random_absG()
                    appG_rand = absG_rand + 5.*np.log10(dist_rand/0.010)
                    if appG_rand>self.appG_low and appG_rand<self.appG_high:
                        parerr_rand = random_parerr_f(appG_rand, b_rand)
                        par_measured = 1./dist_rand + parerr_rand*np.random.normal()
                        z_measured = np.sin(b_rand)/par_measured
                        R_measured = np.cos(b_rand)/par_measured
                        if abs(z_measured)<self.z_lim and R_measured>self.R_cuts[0] and R_measured<self.R_cuts[1]: # distance statement
                            # random brightness here
                            absG_measured = appG_rand + 5.*np.log10(0.010*par_measured)
                            if absG_measured>self.absG_cuts[0] and absG_measured<self.absG_cuts[1]: # brightness statement
                                # apply the open cluster masks
                                outside_OCmask = True
                                if self.maskOCs:
                                    for i_OC in range(self.num_of_OCmasks):
                                        if 180./pi*hp.rotator.angdist([180./pi*l_rand, 180./pi*b_rand], self.OCmasks_centers[i_OC], lonlat=True) < self.OCmasks_radii[i_OC]:
                                            outside_OCmask = False
                                if outside_OCmask:
                                    index_z = int( (z_rand + self.z_high)/z_bin_length + 0.5 )
                                    eff_area[index_z] += completeness(l_rand, b_rand, appG_rand)
            print('time to calc. eff_area:', time.time()-tick0)
            #plt.plot(eff_area_z_vec, eff_area)
            #plt.show()
            return eff_area, eff_area_z_vec