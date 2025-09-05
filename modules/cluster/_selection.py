import numpy as np

def selection(richness, z_obs):

    mask = (richness > 20) * (richness < 200)
    mask *= (z_obs > 0.2) * (z_obs < 1)
    return mask