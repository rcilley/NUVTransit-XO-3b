import numpy as np
import exoplanet as xo
import pymc as pm
import pymc_ext as pmx
import pytensor.tensor as pt
from celerite2.pymc import terms, GaussianProcess

###Will not run without data provided###

#Model:
with pm.Model() as model:
    #Parameter for Stellar density
    log_rho_b = pm.Uniform('log_rho_b', lower=np.log(1), upper=np.log(3), initval=np.log(1.97))#, sigma=0.17/1.97) #assuming 'sd' is sigma
    rho_b = pm.Deterministic('rho_b', pt.exp(log_rho_b))
    
    #Parameter for Rp/Rs, optical
    log_isas_tess_rprs = pm.Uniform('logr_tess_isas', lower=0.5*np.log(30e-6), upper=0.5*np.log(0.2), initval=np.log(0.086))
    isas_tess_rprs = pm.Deterministic('isas_tess_rprs', pt.exp(log_isas_tess_rprs))
    
    #Parameter for Impact parameter
    b = xo.impact_parameter('b', ror=isas_tess_rprs, initval=0.696)
    
    #Constant Limb darkening coeffs
    tess_u = (0.2126, 0.3087)
    
    #Reference midt time
    tess_t0 = pm.Uniform('tess_t0', lower=t0_tess-0.05, upper=t0_tess+0.05, initval=t0_tess)
    
    #Parameter for obital period
    orb_period = pm.Uniform('orb_period', lower=per-(5*0.00014), upper=per+(5*0.00014), initval=per)

    #Set up Orbit
    isas_tess_orbit = xo.orbits.KeplerianOrbit(period=orb_period, t0=tess_t0, b=b, rho_star=rho_b,
                                            ecc=ecc, omega=w)
    #TESS############
    for i in range(globals()['tess_num_transit']):
        tess_time = np.array(globals()['tess_time%s'%i])
        tess_flux = np.array(globals()['tess_flux%s'%i])
        tess_err_flux = np.array(globals()['tess_err%s'%i])
                   
        #Transit model
        def tess_model(t=tess_time):
            tess_lcs = xo.LimbDarkLightCurve(tess_u).get_light_curve(orbit=isas_tess_orbit, r=isas_tess_rprs, t=t)
            tess_lc = pm.Deterministic('tess_lc%d'%i, pm.math.sum(tess_lcs, axis=-1))
            return tess_lc + 1

        #Matern-3/2 GP kernel
        std = MAD(tess_flux)
        tess_logs2 = pm.Uniform('tess_logs2%d'%i, lower=2*np.log(30e-6), upper=2*np.log(1), initval=2*np.log(std))
        tess_GP_log_sigma = pm.Uniform('tess_GP_log_sigma%d'%i, lower=np.log(30e-6), upper=np.log(1), initval=np.log(std))
        tess_GP_log_rho = pm.Uniform('tess_GP_log_rho%d'%i, lower=np.log(1e-3), upper=np.log(1e3), initval=np.log(10.))

        kernel = terms.Matern32Term(sigma=pt.exp(tess_GP_log_sigma), rho=pt.exp(tess_GP_log_rho))
        gp = GaussianProcess(kernel, t=tess_time, diag=pt.exp(tess_logs2), mean=tess_model, quiet=True)

        #Fit model
        gp.marginal('tess_gp_obs%d'%i, observed=tess_flux)
        pm.Deterministic('tess_gp_pred%d'%i, gp.predict(tess_flux))

    #ISAS############
    
    #ISAS transit model
    def isas_model(t=isas_time):
        isas_lcs = xo.LimbDarkLightCurve(tess_u).get_light_curve(orbit=isas_tess_orbit, r=isas_tess_rprs, t=t)
        isas_lc = pm.Deterministic('isas_lc', pm.math.sum(isas_lcs, axis=-1))
        return isas_lc + 1

    #Matern-3/2 GP kernel
    std_isas = MAD(isas_flux)
    isas_logs2 = pm.Uniform('isas_logs2', lower=2*np.log(1e-3), upper=2*np.log(1e3), initval=2*np.log(std_isas))
    isas_GP_log_sigma = pm.Uniform('isas_GP_log_sigma', lower=np.log(1e-3), upper=np.log(1e3), initval=np.log(std_isas))
    isas_GP_log_rho = pm.Uniform('isas_GP_log_rho', lower=np.log(1e-3), upper=np.log(1e3), initval=np.log(10.))

    kernel = terms.Matern32Term(sigma=pt.exp(isas_GP_log_sigma), rho=pt.exp(isas_GP_log_rho))
    gp = GaussianProcess(kernel, t=isas_time, diag=pt.exp(isas_logs2)+isas_err_flux**2, mean=isas_model, quiet=True)

    #Fit model
    pm.Deterministic('isas_gp_pred', gp.predict(isas_flux))
    gp.marginal('isas_gp_obs', observed=isas_flux)
    
    #XMM############
    #Constant limb-darkening coeffs
    xmm_u = (1.553, -0.641)

    #Parameter for XMM mid-t
    xmm_midt = pm.Uniform('xmm_midt', lower=midt -0.05, 
                          upper=midt + 0.07, initval=midt)
    #Parameter for NUV Rp/Rs
    xmm_logrprs = pm.Uniform('xmm_logrprs', lower=0.5*np.log(30e-6), upper=0.5*np.log(0.06), 
                             initval=0.5*np.log(0.01))
    xmm_rprs = pm.Deterministic('xmm_rprs', pt.exp(xmm_logrprs))
    
    #Set up orbit
    xmm_orbit = xo.orbits.KeplerianOrbit(period=orb_period, t0=xmm_midt, b=b, rho_star=rho_b,
                                            ecc=ecc, omega=w)
    
    std = MAD(xmm_flux)
    
    #Transit model
    def xmm_model(t=xmm_time):
        xmm_lcs = xo.LimbDarkLightCurve(xmm_u).get_light_curve(orbit=xmm_orbit, r=xmm_rprs, t=t)
        xmm_lc = pm.Deterministic('xmm_lc', pm.math.sum(xmm_lcs, axis=-1))
        return xmm_lc + 1

    #GP model
    xmm_logs2 = pm.Uniform('xmm_logs2', lower=2*np.log(1e-3), upper=2*np.log(1e3), initval=2*np.log(std))
    xmm_GP_log_sigma = pm.Uniform('xmm_GP_log_sigma', lower=np.log(1e-3), upper=np.log(1e3), initval=np.log(std))
    xmm_GP_log_rho = pm.Uniform('xmm_GP_log_rho', lower=np.log(1e-3), upper=np.log(1e3), initval=np.log(10)) #

    kernel = terms.Matern32Term(sigma=pt.exp(xmm_GP_log_sigma), rho=pt.exp(xmm_GP_log_rho))
    gp = GaussianProcess(kernel, diag=pt.exp(xmm_logs2), mean=xmm_model, quiet=True)
    gp.compute(xmm_time, diag=pt.exp(xmm_logs2)+xmm_err_flux**2, quiet=True)

    #Fit model
    pm.Deterministic('xmm_gp_pred', gp.predict(xmm_flux))
    gp.marginal('xmm_gp_obs', observed=xmm_flux)
        


with model:
    map_soln = pm.find_MAP()
    
    #Run NUTS sampler
    idata = pmx.utils.sample(chains=10, tune=5000, draws=3000, initvals=map_soln, cores=4, target_accept=0.95)
    
#Save results
idata.to_netcdf("XMMTESSISAS_ConstLDC_2025.nc")
