'''
TODO function to find peaks with sigma clipping
TODO function to check the validity of the peaks
  - need a way to verify that the peaks are moving linearly outward from the center
    - reject the peaks that aren't
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
