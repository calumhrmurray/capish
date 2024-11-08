cmd = {}
#cmd['1'] = 'python pinocchio_mcmc.py --type NxMwl --fit_cosmo False --number_params_scaling_relation 4'
#cmd['2'] = 'python pinocchio_mcmc.py --type Mwl --fit_cosmo False --number_params_scaling_relation 4'
#cmd['3'] = 'python pinocchio_mcmc.py --type N --fit_cosmo False --number_params_scaling_relation 4'

#cmd['4'] = 'python pinocchio_mcmc.py --type NxMwl --fit_cosmo True --number_params_scaling_relation 4'
cmd['1'] = 'python pinocchio_mcmc.py --type Mwl --fit_cosmo True --number_params_scaling_relation 4'
#cmd['6'] = 'python pinocchio_mcmc.py --type N --fit_cosmo True --number_params_scaling_relation 4'

import sys, os
code, num = sys.argv
os.system(cmd[str(num)])