import numpy as np
from scipy.integrate import simps

def first_derivative(theta, model, shape_model, delta = 1e-5):
    r"""
    ref : https://en.wikipedia.org/wiki/Finite_difference
    Attributes:
    -----------
    theta : array
        parameter values to evaluate first dreivative
    model : fct
        model to compute second derivative
    Returns:
    --------
    sec : array
        array of first derivative
    """
    first = np.zeros([len(theta)] + list(shape_model))
    for i in range(len(theta)):
        delta_i = np.zeros(len(theta))
        delta_i[i] = delta
        first[i] = (model(theta + delta_i/2) - model(theta - delta_i/2))/delta
    return first

def second_derivative(theta, model, shape_model, delta = 1e-5):
    r"""
    ref : https://en.wikipedia.org/wiki/Finite_difference
    Attributes:
    -----------
    theta : array
        parameter values to evaluate second dreivative
    model : fct
        model to compute second derivative
    Returns:
    --------
    sec : array
        array of second derivative
    """
    sec = np.zeros([len(theta),len(theta)] + list(shape_model))
    for i in range(len(theta)):
        delta_i = np.zeros(len(theta))
        delta_i[i] = delta
        for j in range(len(theta)):
            delta_j = np.zeros(len(theta))
            delta_j[j] = delta
            if i == j:
                sec[i,j] = (model(theta+delta_i) - \
                            2*model(theta) + \
                            model(theta-delta_i))/delta**2
            elif i > j:
                sec[i,j] = (model(theta+delta_i+delta_j) - \
                            model(theta+delta_i-delta_j) - \
                            model(theta-delta_i+delta_j) + \
                            model(theta-delta_i-delta_j))/(4*delta**2)
                sec[j,i] = sec[i,j]
    return sec
    
def Fisher_Matrix(theta, model, inv_cov, delta = 1e-5):
    r"""
    Attributes:
    -----------
    theta : array
        the values of model parameters used to evaluate the Fisher matrix
    model : fct
        the model
    cov : array
        the covariance matrix of the observed data
    Returns:
    --------
    Fisher_matrix : array
        the Fisher matrix for the model parameters
   """
    Fisher_matrix = np.zeros([len(theta), len(theta)])
    shape_model = inv_cov.diagonal().shape
    fd = first_derivative(theta, model, shape_model, delta = 1e-5)
    for i in range(len(theta)):
        for j in range(len(theta)):
            Fisher_matrix[i,j] = np.sum(fd[i]*inv_cov.dot(fd[j]))
    return Fisher_matrix
    
    # def Fisher_Matrix_Gaussian_XY(self, SigmaX, SigmaY_1, d_model, n_dim):
    #     r"""
    #     X: true
    #     Y: input
    #     d_model: model derivatives
        
    #     """
    #     res=np.zeros([n_dim, n_dim])
    #     for i in range(n_dim):
    #         for j in range(n_dim):
    #             res[i,j] = np.sum(d_model[i,:] * np.dot(SigmaY_1, np.dot(SigmaX, SigmaY_1) ).dot(d_model[j,:]))
    #     return res
    
    # def Fisher_Matrix_Gaussian_XX(self, SigmaX_1, d_model, n_dim):
    #     r"""
    #     X: true
    #     Y: input
    #     d_model: model derivatives
        
    #     """
    #     res=np.zeros([n_dim, n_dim])
    #     for i in range(n_dim):
    #         for j in range(n_dim):
    #             res[i,j] = np.sum(d_model[i,:] * SigmaX_1.dot(d_model[j,:]))
    #     return res

    # def S_Fisher_Matrix(self, theta, model, cov, delta = 1e-5):

    #         #https://arxiv.org/pdf/1606.06455.pdf
    #         zeros = np.zeros(len(theta))
    #         first_derivative = []
    #         second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
    #         first_derivative = np.zeros([len(theta), len(cov[:,0])])
    #         def derive_i(theta, i):
    #                     delta_i = np.zeros(len(theta))   
    #                     delta_i[i] = delta
    #                     res = (model(np.array(theta) + delta_i/2) - \
    #                            model(np.array(theta) - delta_i/2))/delta
    #                     return res      
    #         for i in range(len(theta)):
    #             first_derivative[i] = derive_i(theta, i) 
    #             for j in range(len(theta)):
    #                 delta_j = np.zeros(len(theta))
    #                 delta_j[j] = delta
    #                 deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - \
    #                             derive_i(np.array(theta) - delta_j/2, i))/delta
    #                 second_derivative[i,j,:] = deriv_ij
    #         S_matrix = np.zeros([len(theta),len(theta),len(theta)])
    #         for k in range(len(theta)):
    #             for l in range(len(theta)): 
    #                 for m in range(len(theta)):
    #                     S_matrix[k,l,m] = np.sum(second_derivative[k,l]* \
    #                                              np.linalg.inv(cov).dot(first_derivative[m]))
    #         return S_matrix

    # def Q_Fisher_Matrix(self, theta, model, cov, delta = 1e-5):

    #    ## https://arxiv.org/pdf/1606.06455.pdf

    #     zeros = np.zeros(len(theta))
    #     second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
    #     def derive_i(theta, i):
    #                 delta_i = np.zeros(len(theta))
    #                 delta_i[i] = delta
    #                 res = (model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta
    #                 return res
    #     for i in range(len(theta)):
    #         for j in range(len(theta)):
    #             delta_j = np.zeros(len(theta))
    #             delta_j[j] = delta
    #             deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - derive_i(np.array(theta) - delta_j/2, i))/delta
    #             second_derivative[i,j,:] = deriv_ij
    #     Q_matrix = np.zeros([len(theta),len(theta),len(theta), len(theta)])
    #     for k in range(len(theta)):
    #         for l in range(len(theta)): 
    #             for m in range(len(theta)):
    #                 for n in range(len(theta)):
    #                     Q_matrix[k,l,m,n] = np.sum(second_derivative[k,l]*np.linalg.inv(cov).dot(second_derivative[m,n]))
    #     return Q_matrix
    #     return lnL_Gaussian + lnL_Q_Gaussian + lnL_S_Gaussian

    # def lnL_Fisher(self, theta, MLE, Fisher_matrix):
    #     dtheta = theta - MLE
    #     lnL_Gaussian = -0.5*np.sum(dtheta*Fisher_matrix.dot(dtheta))
    #     return lnL_Gaussian 

    # def lnL_S_Fisher(self, theta, MLE, S_matrix):
    #     dtheta = theta - MLE
    #     S = 0
    #     for i in range(len(theta)):
    #         for j in range(len(theta)): 
    #             for k in range(len(theta)):
    #                 S += S_matrix[i,j,k]*dtheta[i]*dtheta[j]*dtheta[k]
    #     lnL_S_Gaussian = -(1./2.)*S

    #     return lnL_S_Gaussian

    # def lnL_Q_Fisher(self, theta, MLE, Q_matrix):
    #     dtheta = theta - MLE
    #     Q = 0
    #     for i in range(len(theta)):
    #         for j in range(len(theta)): 
    #             for k in range(len(theta)):
    #                 for l in range(len(theta)):
    #                         Q += Q_matrix[i,j,k,l]*dtheta[i]*dtheta[j]*dtheta[k]*dtheta[l]
    #     lnL_Q_Gaussian = -(1./8.)*Q

    #     return lnL_Q_Gaussian
    
#     def Fisher_matrix_unbinned_Poissonian(self, theta, Z_bin, logMass_bin, CA):
#         r"""
#         z_min, z_max = Z_bin
#         logm_min, logm_max = logMass_bin
#         """
#         z_min, z_max = Z_bin
#         logm_min, logm_max = logMass_bin
        
#         def model_Ntot(theta):
#             Omegab = 0.048254
#             Omegam, sigma8 = theta
#             cosmo_new = ccl.Cosmology(Omega_c = Omegam - Omegab, Omega_b = Omegab, h = 0.6777, sigma8 = sigma8, n_s=0.96)
#             massdef_new = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
#             hmd_new = ccl.halos.MassFuncDespali16(cosmo_new, mass_def=massdef_new)
#             CA.set_cosmology(cosmo = cosmo_new, hmd = hmd_new, massdef = massdef_new)
#             CA.compute_multiplicity_grid_MZ(z_grid = CA.z_grid, logm_grid = CA.logm_grid)
#             N_th = CA.Cluster_Abundance_MZ(Redshift_bin = [[z_min, z_max]], 
#                                            Proxy_bin = [[logm_min, logm_max]], method = 'simps')
#             return N_th[0][0]

#         def model_grid_ln(theta):
#             Omegab = 0.048254
#             Omegam, sigma8 = theta
#             cosmo_new = ccl.Cosmology(Omega_c = Omegam - Omegab, Omega_b = Omegab, h = 0.6777, sigma8 = sigma8, n_s=0.96)
#             massdef_new = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
#             hmd_new = ccl.halos.MassFuncDespali16(cosmo_new, mass_def=massdef_new)
#             CA.set_cosmology(cosmo = cosmo_new, hmd = hmd_new, massdef = massdef_new)
#             CA.compute_multiplicity_grid_MZ(z_grid = CA.z_grid, logm_grid = CA.logm_grid)
#             return np.log(CA.sky_area * CA.dN_dzdlogMdOmega)

#         N_th_cosmo_true_unbinned = model_Ntot(theta)
#         def av_ln_multiplicity_n(theta, model, delta = 1e-5):
#             model_true = np.exp(model(theta))
#             pdf = model_true
#             index_z_grid = np.arange(len(CA.z_grid))
#             index_logm_grid = np.arange(len(CA.logm_grid))
#             mask_z = (CA.z_grid > z_min)*(CA.z_grid < z_max)
#             mask_logm = (CA.logm_grid > logm_min)*(CA.logm_grid < logm_max)
#             index_z_mask = index_z_grid[mask_z]
#             index_logm_mask = index_logm_grid[mask_logm]
#             res = np.zeros([len(theta),len(theta)])
#             sec_derivative = second_derivative(theta, model_grid_ln, model_true.shape, delta = delta)
#             for i in range(len(theta)):
#                 for j in range(len(theta)):
#                     if i >= j:
#                         integrand = sec_derivative[i,j] * pdf
#                         integrand_cut = np.array([integrand[:,i][mask_logm] for i in index_z_mask])
#                         res[i,j] = simps(simps(integrand_cut, CA.logm_grid[mask_logm]), CA.z_grid[mask_z])
#                         res[j,i] = res[i,j]
#             return  res 

#         Ntot_second_derivative = second_derivative(theta, model_Ntot, N_th_cosmo_true_unbinned.shape, delta = 1e-4)
#         av_ln_lambda = av_ln_multiplicity_n(theta, model_grid_ln, delta = 1e-4)
#         Fisher_unBinned_Poissonian = Ntot_second_derivative - av_ln_lambda
#         #cov_param_unBinned_Poissonian = np.linalg.inv(Fisher_unBinned_Poissonian)
#         return Fisher_unBinned_Poissonian

# def cov_Frequentist(cov_Bayesian, d_model, Sigma):
#     r"""
#     compute frequentist covariance matrix
#     Attributes:
#     -----------
#     cov_Bayesian: Bayesian covariance matrix
#     d_model: 
#     """
#     cov_freq=np.zeros([2,2])
#     for i in range(2):
#         for j in range(2): 
#             aT=0
#             for k in range(2): aT+=cov_Bayesian[i,k]*d_model[k]
#             a=0
#             for k in range(2): a+=cov_Bayesian[j,k]*d_model[k]
#             cov_freq[i,j] = np.sum(aT.flatten()*Sigma.dot(a.flatten()))
#     return cov_freq


