'''
This program is part a of Bachelor's Thesis for Biotechnology and
Biomedical Engineering. The program uses open access dataset
(‘The SWEC-ETHZ iEEG Database and Algorithms’.
Accessed: Jan. 19, 2024. [Online]. Available: http://ieeg-swez.ethz.ch/). The
data is divided to segments of 3 minute pre-ictal, x- time ictal and 3 minute
post-ictal data.

Functionality:
 The following program uses multiple funtions to derive the PSD ratios from raw
 EEG data. It saves them to folder for later assesments. Futhermore, it reads
 the data from the file to form ML models from them and writes the accuracy to
 excvel files.
 There are parts of the functions unused and some parts of the function calls
 commented out. These can be used to analyse the data futher. The model now
 formulates model for all patients with >= 7 seizures and one model to all.


Programs writer
 * Name: Markus Haukipää
 * E-Mail: markus.haukipaa@tuni.fi

'''
import math
import os

import mne
import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn import neighbors
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import json
import pandas as pd


#defined frequency bands
delta_band = [0.5,4]
theta_band = [4,8]
alpha_band = [8,13]
beta_band = [13,30]
low_gamma_band = [30,80]
high_gamma_band = [80,145] # 150 has interference harmonics from line current

# The files divided to a and b subclasses were combined to one file by the number
fileNumber = ['1','2','3','4','5','6','7','8','9','10',
              '11','12','13','14','15','16']
fs = 512
PreICTAL, ICTAL, PostICTAL = -1,0,1

# SeizureTotalCount = 64

# A Dict construct of learning and test data.
# The data is divided by Ictal phases
# Futhermore divided into 14 ratios between the frequency powers
'''ratios = {
    'learning':{
        PreICTAL: {
            'delta': {
                'theta': [],
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'theta': {
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'alpha': {
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'beta': {
                'low_gamma': [],
                'high_gamma': []
            },
            'low_gamma': {
                'high_gamma': []
            }
        },
        ICTAL: {
            'delta': {
                'theta': [],
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'theta': {
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'alpha': {
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'beta': {
                'low_gamma': [],
                'high_gamma': []
            },
            'low_gamma': {
                'high_gamma': []
            }
        },
        PostICTAL: {
            'delta': {
                'theta': [],
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'theta': {
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'alpha': {
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'beta': {
                'low_gamma': [],
                'high_gamma': []
            },
            'low_gamma': {
                'high_gamma': []
            }
        }
    },
    'test':{
        PreICTAL: {
            'delta': {
                'theta': [],
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'theta': {
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'alpha': {
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'beta': {
                'low_gamma': [],
                'high_gamma': []
            },
            'low_gamma': {
                'high_gamma': []
            }
        },
        ICTAL: {
            'delta': {
                'theta': [],
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'theta': {
                'alpha': [],
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'alpha': {
                'beta': [],
                'low_gamma': [],
                'high_gamma': []
            },
            'beta': {
                'low_gamma': [],
                'high_gamma': []
            },
            'low_gamma': {
                'high_gamma': []
            }
        },
        PostICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                }
            }
    }
}'''
importData = {
    'delta': {
        'theta': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'alpha': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'beta': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'low_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'high_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
    },
    'theta': {
        'alpha': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'beta': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'low_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'high_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
    },
    'alpha': {
        'beta': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'low_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'high_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
    },
    'beta': {
        'low_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
        'high_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
    },
    'low_gamma': {
        'high_gamma': {
            'ID1': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID4': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID5': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID9': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID12': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID13': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID14': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            },
            'ID0': {
                'preIctal': None,
                'preIctal+postIctal': None,
                'Ictal': None,
                'postIctal': None,
                'postIctal + preIctal': None
            }
        },
    }
}

#fn = 264 and the lowest frequency measured is 150

def main():

    #Form the ratio files and preprocess
    PreProcessAll()
    PreProcessPatient()

    # If need to test the Cherrypicked performance
    #CherryPickID1()
    # Load the data from files PreProcess dowloaded them. Stucture equals to
    # dict ratios

    ID1 = readFile('RatioFile/PSD/ID1')
    ID4 = readFile('RatioFile/PSD/ID4')
    ID5 = readFile('RatioFile/PSD/ID5')
    ID9 = readFile('RatioFile/PSD/ID9')
    ID12 = readFile('RatioFile/PSD/ID12')
    ID13 = readFile('RatioFile/PSD/ID13')
    ID14 = readFile('RatioFile/PSD/ID14')
    IDALL = readFile('RatioFile/PSD/ALL')

    # import data to excel as dict
    patientList = [ID1, ID4, ID5, ID9, ID12, ID13, ID14, IDALL]

    num = [1, 4, 5, 9, 12, 13, 14, 0]

    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }

    #KNearestNeighbour
    for patient, number in zip(patientList, num):
        importData = LearnAndTestBinary(patient, number,'KNN',importData)
        WriteToExcel('KNN_4_Uni_V_Bi_perf',importData)

    # SupportVectorMachine
    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }

    for patient, number in zip(patientList, num):
        importData =  LearnAndTestBinary(patient, number,'SVM',importData)
        WriteToExcel('SVM_5_linear_V_Bi_perf',importData)

    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }

    #RandomForest
    for patient, number in zip(patientList, num):
        importData = LearnAndTestBinary(patient, number, 'RF',importData)
        WriteToExcel('RF_200_V_Bi_perf',importData)

    #Cherry-picked
    '''ID1Cherry = readFile('RatioFile/PSD/CherryPickedID1')
    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }

    # SupportVectorMachine
    importData = LearnAndTestBinary(ID1Cherry, 1, 'SVM', importData)
    WriteToExcel('SVM_5_linear_V_CP2_perf', importData)

    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }

    # KNearestNeighbour

    importData = LearnAndTestBinary(ID1Cherry, 1, 'KNN', importData)
    WriteToExcel('KNN_4_Uni_V_CP2_perf', importData)

    importData = {
        'delta': {
            'theta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'all': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'theta': {
            'alpha': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'alpha': {
            'beta': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'beta': {
            'low_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'low_gamma': {
            'high_gamma': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            },
        },
        'delta_theta': {
            'all_other': {
                'ID1': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID4': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID5': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID9': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID12': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID13': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID14': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                },
                'ID0': {
                    'preIctal': None,
                    'preIctal+postIctal': None,
                    'Ictal': None,
                    'postIctal': None,
                    'postIctal + preIctal': None
                }
            }
        }
    }'''

    # RandomForest
    '''importData = LearnAndTestBinary(ID1Cherry, 1, 'RF', importData)
    WriteToExcel('RF_200_V_CP_perf', importData)'''
    ''' importData = {
            'delta': {
                'theta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'alpha': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'all': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'theta': {
                'alpha': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'alpha': {
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'beta': {
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'low_gamma': {
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'delta_theta': {
                'all_other': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                }
            }
        }'''

    '''# SVM
    importData = LearnAndTestBinary(ID1Cherry, 1, 'SVM', importData)
    WriteToExcel('SVM_V_CP_perf', importData)'''

    ''' importData = {
            'delta': {
                'theta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'alpha': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'all': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'theta': {
                'alpha': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'alpha': {
                'beta': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'beta': {
                'low_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'low_gamma': {
                'high_gamma': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                },
            },
            'delta_theta': {
                'all_other': {
                    'ID1': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID4': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID5': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID9': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID12': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID13': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID14': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    },
                    'ID0': {
                        'preIctal': None,
                        'preIctal+postIctal': None,
                        'Ictal': None,
                        'postIctal': None,
                        'postIctal + preIctal': None
                    }
                }
            }
        }'''
    # RandomForest
    '''importData = LearnAndTestBinary(ID1Cherry, 1, 'KNN', importData)
    WriteToExcel('KNN_V_CP_perf', importData)'''



def LearnAndTest(patient, number,classifier,importData):
    '''Used as a intermediate function to uphold clear fctionality of other
         functions. calls wanted Machine learning funtion by the different
          ratios
          :return dict (of the updated importData)'''
    print('Delta/Theta')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'theta'], number,classifier,importData)

    print('Delta/Alpha')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'alpha'], number,classifier,importData)

    print('Delta/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'beta'], number,classifier,importData)

    print('Delta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'low_gamma'], number,classifier,importData)

    print('Delta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'high_gamma'], number,classifier,importData)

    print('Delta/All')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta', 'all'], number,
                                 classifier, importData)

    print("#################################################")

    print('Theta/Alpha')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['theta', 'alpha'], number,classifier,importData)

    print('Theta/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['theta', 'beta'], number,classifier,importData)

    print('Theta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['theta', 'low_gamma'], number,classifier,importData)

    print('Theta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['theta', 'high_gamma'], number,classifier,importData)

    print('Alpha/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['alpha', 'beta'], number,classifier,importData)

    print('Alpha/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['alpha', 'low_gamma'], number,classifier,importData)

    print('Alpha/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['alpha', 'high_gamma'], number,classifier,importData)

    print('Beta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['beta', 'low_gamma'], number,classifier,importData)

    print('Beta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['beta', 'high_gamma'], number,classifier,importData)

    print('Low_gamma/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['low_gamma', 'high_gamma'], number,classifier,importData)


    print('delta_theta/all_other')
    print('Patient number: ' + str(number))
    importData = MachineLearning(patient, ['delta_theta', 'all_other'], number,
                                 classifier, importData)


    return importData
def LearnAndTestBinary(patient, number,classifier,importData):
    '''Used as a intermediate function to uphold clear fctionality of other
     functions. calls wanted Machine learning funtion by the different
      ratios
      :return dict (of the updated importData)'''
    print('Delta/Theta')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'theta'], number,classifier,importData)

    print('Delta/Alpha')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'alpha'], number,classifier,importData)

    print('Delta/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'beta'], number,classifier,importData)

    print('Delta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'low_gamma'], number,classifier,importData)

    print('Delta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'high_gamma'], number,classifier,importData)

    print('Delta/All')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta', 'all'], number,
                                 classifier, importData)

    print("#################################################")

    print('Theta/Alpha')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['theta', 'alpha'], number,classifier,importData)

    print('Theta/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['theta', 'beta'], number,classifier,importData)

    print('Theta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['theta', 'low_gamma'], number,classifier,importData)

    print('Theta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['theta', 'high_gamma'], number,classifier,importData)

    print('Alpha/Beta')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['alpha', 'beta'], number,classifier,importData)

    print('Alpha/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['alpha', 'low_gamma'], number,classifier,importData)

    print('Alpha/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['alpha', 'high_gamma'], number,classifier,importData)

    print('Beta/Low_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['beta', 'low_gamma'], number,classifier,importData)

    print('Beta/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['beta', 'high_gamma'], number,classifier,importData)

    print('Low_gamma/High_gamma')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['low_gamma', 'high_gamma'], number,classifier,importData)


    print('delta_theta/all_other')
    print('Patient number: ' + str(number))
    importData = MachineLearningBinary(patient, ['delta_theta', 'all_other'], number,
                                 classifier, importData)


    return importData
def WriteToExcel(title,importData):
    df = pd.DataFrame.from_dict({(i, j, z): importData[i][j][z]
                                 for i in importData.keys()
                                 for j in importData[i].keys()
                                 for z in importData[i][j].keys()},
                                orient='index')

    df.to_excel(title+'.xlsx')
    return
def CherryPickID1():
    ratios = {
        'learning': {
            PreICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            ICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            PostICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            }
        },
        'test': {
            PreICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            ICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            PostICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            }
        }
    }
    for Sz_number in range (13):

        EEG = loadData('iEEG/ID1/Sz' + str(Sz_number + 1) + '.mat')
        T = len(EEG)  # sampling points
        M = len(EEG[0])  # number of electrodes
        time = T / fs
        step = 1 / fs
        freq = np.arange(0, time, step)

        #dividing test and learn approx 80% (and 20 % testing)
        # 29 Sezure channels, 2-3 per seizure
        if Sz_number == 0:
            for channel in [4,15,26]:
                ratios = updateRatios(EEG[:,channel],fs,ratios,'learning')
            '''basicPlot(EEG[:, 26], freq, time)
            basicPlot(EEG[:, 15], freq, time)
            basicPlot(EEG[:, 4], freq, time)'''

        if Sz_number == 1:
            for channel in [26, 27, 34]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 26], freq, time)
            basicPlot(EEG[:, 27], freq, time)
            basicPlot(EEG[:, 34], freq, time)'''

        if Sz_number == 2:
            for channel in [26, 28, 29]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 26], freq, time)
            basicPlot(EEG[:, 28], freq, time)
            basicPlot(EEG[:, 29], freq, time)'''

        if Sz_number == 3:
            for channel in [18, 39]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            ''' basicPlot(EEG[:, 18], freq, time)
             basicPlot(EEG[:, 39], freq, time)'''
        if Sz_number == 4:
            for channel in [23, 26, 29]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            ''' basicPlot(EEG[:, 23], freq, time)
             basicPlot(EEG[:, 26], freq, time)#++
             basicPlot(EEG[:, 29], freq, time)#++++'''
        if Sz_number == 5:
            for channel in [17, 18, 19]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 17], freq, time)
            basicPlot(EEG[:, 18], freq, time)
            basicPlot(EEG[:, 19], freq, time)'''
        if Sz_number == 6:
            for channel in [2,21]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            ''' basicPlot(EEG[:, 21], freq, time)
            basicPlot(EEG[:, 2], freq, time)'''

        if Sz_number == 7:
            for channel in [1,4,29]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            ''' basicPlot(EEG[:, 1], freq, time)
            basicPlot(EEG[:, 4], freq, time)
            basicPlot(EEG[:, 29], freq, time)'''

        if Sz_number == 8: #artifact at 65-70
            for channel in [4, 15, 26]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 18], freq, time)
            basicPlot(EEG[:, 33], freq, time)#
            basicPlot(EEG[:, 39], freq, time)##'''

        if Sz_number == 9:
            for channel in [3, 4]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 3], freq, time)
            basicPlot(EEG[:, 4], freq, time)'''

        if Sz_number == 12:
            for channel in [23,26]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'learning')
            '''basicPlot(EEG[:, 23], freq, time)
            basicPlot(EEG[:, 26], freq, time)'''
        #*********TEST FILES*********
        #2-3 per seizure, total of 6
        if Sz_number == 10:
            for channel in [4, 15, 26]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'test')
            '''basicPlot(EEG[:, 16], freq, time)
            basicPlot(EEG[:, 17], freq, time)
            #basicPlot(EEG[:, 26], freq, time)'''
        if Sz_number == 11:
            for channel in [2, 3, 31]:
                ratios = updateRatios(EEG[:, channel], fs, ratios,'test')
            '''#basicPlot(EEG[:, 0], freq, time)
            basicPlot(EEG[:, 2], freq, time)
            basicPlot(EEG[:, 3], freq, time)
            basicPlot(EEG[:, 31], freq, time)'''
        # **********************

    #writeToFile('RatioFile/PSD/CherryPickedID1', ratios)
def plotAll():
    '''Function for plotting all seizures and channels for visual expection'''
    for file in fileNumber:
        folder = 'iEEG/ID' + file
        count = 0

        # number of files
        for filePath in os.listdir(folder):
            count += 1
        # Go through patient files

        for Sz_number in range(count):


            EEG = loadData(folder + '/Sz' + str(Sz_number+1) + '.mat')
            T = len(EEG)  # sampling points
            M = len(EEG[0])  # number of electrodes
            time = T / fs
            step = 1 / fs
            freq = np.arange(0, time, step)
            # plot electrodes to same plot
            print(str(file) + " adn " +str(Sz_number) )
            for electrode in range(M):
                #for electrode in range(M):
                print("newPlot")
                print("electrode index = " + str(electrode))
                basicPlot(EEG[:, electrode], freq, time)
def updateRatios(EEG,fs,ratios,type):
    '''Funtion used to update the importData file with new ratio values
    :return dict (uptaded ratios)'''
    T= len(EEG)
    time = T / fs
    step = 1 / fs
    freq = np.arange(0, time, step)
    filterdEEG = mne.filter.notch_filter(EEG,fs, (50, 2 * 50))

    # Vizulation of the data in Subbands
    delta, theta, alpha, beta, low_gamma, high_gamma \
        = toBands(filterdEEGs, freq, time)

    three_minute = 3 * 60
    # name the data through preIctal,ictal and postIctal

    preIcatalEEG = filterdEEG[0:(three_minute * fs)]
    IctalEEG = filterdEEG[(three_minute * fs):
                          (T - three_minute * fs)]
    postIctalEEG = filterdEEG[(T - three_minute * fs):T]

    label = PreICTAL  # starting label

    for data in [preIcatalEEG, IctalEEG, postIctalEEG]:

        # Attrieving the power spectral eeg data from the bands
        PSDdelta, PSDtheta, PSDalpha, PSDbeta, PSDlow_gamma,\
        PSDhigh_gamma = getPower(freq, data)

        # Making the power spectral ratios between different bands
        # Feature extraction
        ratios = PlaceToDict(
            ratios,
            type,
            label,
            PSDhigh_gamma,
            PSDlow_gamma,
            PSDbeta,
            PSDalpha,
            PSDtheta,
            PSDdelta)
        label += 1

    return ratios
def PreProcessAll():
    '''Preprocess all the data, divide to test and learning data and
     save ratios of all patients to a file '''
    ''' Preprocess all files with filtering and write a power
       spectral density ratio file. The the PSD ratio is made from dividing
       between the different frequency bands'''
    ratios = {
        'learning': {
            PreICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            ICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            PostICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            }
        },
        'test': {
            PreICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            ICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            },
            PostICTAL: {
                'delta': {
                    'theta': [],
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': [],
                    'all': []
                },
                'theta': {
                    'alpha': [],
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'alpha': {
                    'beta': [],
                    'low_gamma': [],
                    'high_gamma': []
                },
                'beta': {
                    'low_gamma': [],
                    'high_gamma': []
                },
                'low_gamma': {
                    'high_gamma': []
                },
                'delta_theta': {
                    'all_other': []
                }
            }
        }
    }
    for file in fileNumber:
        folder = 'iEEG/ID' + file
        count = 0

        # number of files
        for filePath in os.listdir(folder):
            count += 1
        # Go through patient files

        for Sz_number in range(count):

            EEG = loadData(folder + '/Sz' + str(Sz_number + 1) + '.mat')
            T = len(EEG)  # sampling points
            M = len(EEG[0])  # number of electrodes
            time = T / fs
            step = 1 / fs
            freq = np.arange(0, time, step)
            # plot electrodes to same plot

            for electrode in range(M):

                # filtering line current noise, 50 Hz and harmonics
                filterdEEG = mne.filter.notch_filter(EEG[:, electrode],
                                                     fs, (50, 2 * 50))

                three_minute = 3 * 60
                # name the data through preIctal,ictal and postIctal

                preIcatalEEG = filterdEEG[0:(three_minute * fs)]
                IctalEEG = filterdEEG[(three_minute * fs):
                                      (T - three_minute * fs)]
                postIctalEEG = filterdEEG[(T - three_minute * fs):T]

                label = PreICTAL  # starting label

                for data in [preIcatalEEG, IctalEEG, postIctalEEG]:

                    #Vizulation of the data in Subbands
                    '''delta, theta, alpha, beta, low_gamma, high_gamma \
                                                = toBands(data, freq, time)'''

                    # Attrieving the power spectral eeg data from the bands
                    PSDdelta, PSDtheta, PSDalpha, PSDbeta, PSDlow_gamma, \
                    PSDhigh_gamma = getPower(freq, data)

                    # Making the power spectral ratios between different bands
                    # Feature extraction

                    # Training file fro 80% of the data
                    if (Sz_number < count * 0.8):
                        ratios = PlaceToDict(
                            ratios,
                            'learning',
                            label,
                            PSDhigh_gamma,
                            PSDlow_gamma,
                            PSDbeta,
                            PSDalpha,
                            PSDtheta,
                            PSDdelta)
                        label += 1


                    # Make the test set of the approx. 20%
                    else:
                        ratios = PlaceToDict(
                            ratios,
                            'test',
                            label,
                            PSDhigh_gamma,
                            PSDlow_gamma,
                            PSDbeta,
                            PSDalpha,
                            PSDtheta,
                            PSDdelta)
                        label += 1
    writeToFile('RatioFile/PSD/ALL', ratios)
def PreProcessPatient():
    '''Preprocess patients the data with >= 7 seizures,
        divide to test and learning data and
         save ratios of a patient to a file. This is done for all the data '''
    ''' Preprocess patient files with filtering and write a power
    spectral density ratio file. The the PSD ratio is made from dividing
    between the different frequency bands'''
    for file in fileNumber:
        ratios = {
            'learning': {
                PreICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                },
                ICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                },
                PostICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                }
            },
            'test': {
                PreICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                },
                ICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                },
                PostICTAL: {
                    'delta': {
                        'theta': [],
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': [],
                        'all': []
                    },
                    'theta': {
                        'alpha': [],
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'alpha': {
                        'beta': [],
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'beta': {
                        'low_gamma': [],
                        'high_gamma': []
                    },
                    'low_gamma': {
                        'high_gamma': []
                    },
                    'delta_theta': {
                        'all_other': []
                    }
                }
            }
        }
        folder = 'iEEG/ID' + file
        count = 0

        # number of files
        for filePath in os.listdir(folder):
            count += 1
        # Go through patient files

        if count >= 7:  # neglect the files under the learning limit (7)
            for Sz_number in range(count):

                EEG = loadData(folder + '/Sz' + str(Sz_number + 1) + '.mat')
                T = len(EEG)  # sampling points
                M = len(EEG[0])  # number of electrodes
                time = T / fs
                step = 1 / fs
                freq = np.arange(0, time, step)

                # plot electrodes to same plot
                for electrode in range(M):

                    # filtering line current noise, 50 Hz and harmonics
                    filterdEEG = mne.filter.notch_filter(EEG[:, electrode],
                                                         fs, (50, 2 * 50))
                    #toBands(filterdEEG)

                    three_minute = 3 * 60

                    # name the data through preIctal,ictal and postIctal

                    preIcatalEEG = filterdEEG[0:(three_minute * fs)]
                    IctalEEG = filterdEEG[(three_minute * fs):
                                          (T - three_minute * fs)]
                    postIctalEEG = filterdEEG[(T - three_minute * fs):T]

                    label = PreICTAL  # starting label

                    for data in [preIcatalEEG, IctalEEG, postIctalEEG]:

                        # Attrieving the power spectral eeg data from the bands
                        PSDdelta, PSDtheta, PSDalpha, PSDbeta, PSDlow_gamma, \
                        PSDhigh_gamma = getPower(freq, data)

                        # Making the power spectral ratios between different bands
                        # Feature extraction

                        # Training file fro 80% of the data
                        if (Sz_number < count * 0.8):
                            ratios = PlaceToDict(
                                ratios,
                                'learning',
                                label,
                                PSDhigh_gamma,
                                PSDlow_gamma,
                                PSDbeta,
                                PSDalpha,
                                PSDtheta,
                                PSDdelta)
                            label += 1


                        # Make the test set of the approx. 20%
                        else:
                            ratios = PlaceToDict(
                                ratios,
                                'test',
                                label,
                                PSDhigh_gamma,
                                PSDlow_gamma,
                                PSDbeta,
                                PSDalpha,
                                PSDtheta,
                                PSDdelta)
                            label += 1
            writeToFile('RatioFile/PSD/ID' + file, ratios)
def PlaceToDict(ratios, datatype, label, PSDhigh_gamma, PSDlow_gamma,
                PSDbeta, PSDalpha, PSDtheta, PSDdelta):
    '''Places the ratio between 2 sub bands (feature) to a dictonary
     Total of 14 categories
     :return dict (of updated ratios)'''

    ratios[datatype][label]['delta']['theta']. \
        append(PSDdelta / PSDtheta)

    ratios[datatype][label]['delta']['alpha']. \
        append(PSDdelta / PSDalpha)

    ratios[datatype][label]['delta']['beta']. \
        append(PSDdelta / PSDbeta)

    ratios[datatype][label]['delta']['low_gamma']. \
        append(PSDdelta / PSDlow_gamma)

    ratios[datatype][label]['delta']['high_gamma']. \
        append(PSDdelta / PSDhigh_gamma)

    ratios[datatype][label]['delta']['all']. \
        append(PSDdelta / (PSDhigh_gamma+PSDlow_gamma+PSDalpha+
                           PSDbeta+PSDtheta))

    ratios[datatype][label]['theta']['alpha']. \
        append(PSDtheta / PSDalpha)

    ratios[datatype][label]['theta']['beta']. \
        append(PSDtheta / PSDbeta)

    ratios[datatype][label]['theta']['low_gamma']. \
        append(PSDtheta / PSDlow_gamma)

    ratios[datatype][label]['theta']['high_gamma']. \
        append(PSDtheta / PSDhigh_gamma)

    ratios[datatype][label]['alpha']['beta']. \
        append(PSDalpha / PSDbeta)

    ratios[datatype][label]['alpha']['low_gamma']. \
        append(PSDalpha / PSDlow_gamma)

    ratios[datatype][label]['alpha']['high_gamma']. \
        append(PSDalpha / PSDhigh_gamma)

    ratios[datatype][label]['beta']['low_gamma']. \
        append(PSDbeta / PSDlow_gamma)

    ratios[datatype][label]['beta']['high_gamma']. \
        append(PSDbeta / PSDhigh_gamma)

    ratios[datatype][label]['low_gamma']['high_gamma']. \
        append(PSDlow_gamma / PSDhigh_gamma)

    ratios[datatype][label]['delta_theta']['all_other']. \
        append((PSDdelta+PSDtheta) / (PSDhigh_gamma + PSDlow_gamma + PSDalpha +
                           PSDbeta))

    return ratios
def MachineLearning(Data, ratio, number,classifier,importData):
    '''Learns and tests chosen machine learning model in
     3-class classiification model, in addition
    calculates classsification accuracy
     :return dict, importData where the results are distributed'''

    # Machine learning with supervised support vector machine
    if classifier == 'SVM':
        clf = sk.svm.SVC(decision_function_shape='ovr',kernel='linear')

    elif classifier == 'KNN':
        clf = sk.neighbors.KNeighborsClassifier(n_neighbors=4,
                                                weights='uniform')
    elif classifier == 'RF':
        clf = sk.ensemble.RandomForestClassifier(n_estimators=200)
    # classifier with one vs all

    print("Learning data atrival")
    preIctalPSD = np.array(Data['learning'][PreICTAL][ratio[0]][ratio[1]])
    ictalPSD = np.array(Data['learning'][ICTAL][ratio[0]][ratio[1]])
    postIctalPSD = np.array(Data['learning'][PostICTAL][ratio[0]][ratio[1]])

    print("Test data atrival:")
    TESTpreIctalPSD = np.array(Data['test'][PreICTAL][ratio[0]][ratio[1]])
    TESTictalPSD = np.array(Data['test'][ICTAL][ratio[0]][ratio[1]])
    TESTpostIctalPSD = np.array(Data['test'][PostICTAL][ratio[0]][ratio[1]])

    # learning data
    X = np.concatenate((preIctalPSD, ictalPSD, postIctalPSD))

    # normalize data to increase processing time
    TestMax = np.concatenate((TESTpreIctalPSD, TESTictalPSD,
                              TESTpostIctalPSD)).max()

    MaxValue = X.max()

    if TestMax > X.max():
        MaxValue = TestMax


    X = np.divide(X,MaxValue)

    # [n_samples] labeling all the features
    Y = np.array([PreICTAL] * len(preIctalPSD) + [ICTAL] * len(preIctalPSD)
                 + [PostICTAL] * len(postIctalPSD))

    # reshape to match the svm model
    # [n_samples, n_classes]
    X = X.reshape(-1, 1)

    print("Fitting the learning data")
    clf.fit(X, Y)

    # test data

    # normalize data
    TESTpreIctalPSD = np.divide(TESTpreIctalPSD.reshape(-1, 1), MaxValue)
    TESTictalPSD = np.divide(TESTictalPSD.reshape(-1, 1), MaxValue)
    TESTpostIctalPSD = np.divide(TESTpostIctalPSD.reshape(-1, 1), MaxValue)

    y = clf.predict(TESTpreIctalPSD)
    y1 = clf.predict(TESTictalPSD)
    y2 = clf.predict(TESTpostIctalPSD)

    y.reshape(-1, 1)
    y1.reshape(-1, 1)
    y2.reshape(-1, 1)

    correct = np.where(y == [PreICTAL])
    partlycorrect = np.where(y == [PostICTAL])

    correct1 = np.where(y1 == [ICTAL])

    correct2 = np.where(y2 == [PostICTAL])
    partlycorrect2 = np.where(y2 == [PreICTAL])

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['preIctal'] \
        = correct[0].size / y.size

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['preIctal+postIctal'] \
        = (correct[0].size + partlycorrect[0].size) / y.size

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['Ictal'] = \
        correct1[0].size / y1.size

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['postIctal'] \
        = correct2[0].size / y2.size

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['postIctal + preIctal'] \
        = (correct2[0].size + partlycorrect2[0].size) / y2.size

    '''print('Preictal accuracy: ' + str(correct[0].size/y.size))

    print('Preictal + PostIctal Ictal accuracy: '+
          str((correct[0].size+partlycorrect[0].size)/y.size))

    print('Ictal accuracy: ' + str(correct1[0].size/y1.size))

    print('Postictal accuracy: ' + str(correct2[0].size/y2.size))

    print('PostIctal + PreIctal accuracy: ' +
          str((correct2[0].size + partlycorrect2[0].size) / y2.size))'''

    return importData
def MachineLearningBinary(Data, ratio, number,classifier,importData):
    '''Learns and tests chosen machine learning model in
         2-class classiification model, in addition
        calculates classsification accuracy
        :return dict, importData where the results are distributed '''
    # Machine learning with supervised support vector machine
    if classifier == 'SVM':
        clf = sk.svm.SVC(decision_function_shape='ovr',kernel='linear')

    elif classifier == 'KNN':
        clf = sk.neighbors.KNeighborsClassifier(n_neighbors=4,
                                                weights='uniform')
    elif classifier == 'RF':
        clf = sk.ensemble.RandomForestClassifier(n_estimators=200)
    # classifier with one vs all

    print("Learning data atrival")
    preIctalPSD = np.array(Data['learning'][PreICTAL][ratio[0]][ratio[1]])
    ictalPSD = np.array(Data['learning'][ICTAL][ratio[0]][ratio[1]])

    print("Test data atrival:")
    TESTpreIctalPSD = np.array(Data['test'][PreICTAL][ratio[0]][ratio[1]])
    TESTictalPSD = np.array(Data['test'][ICTAL][ratio[0]][ratio[1]])

    # learning data
    X = np.concatenate((preIctalPSD, ictalPSD))
    print(len(X))
    # normalize data to increase processing time
    TestMax = np.concatenate((TESTpreIctalPSD, TESTictalPSD)).max()

    MaxValue = X.max()

    if TestMax > X.max():
        MaxValue = TestMax


    X = np.divide(X,MaxValue)

    # [n_samples] labeling all the features
    Y = np.array([PreICTAL] * len(preIctalPSD) + [ICTAL] * len(preIctalPSD))

    # reshape to match the svm model
    # [n_samples, n_classes]
    X = X.reshape(-1, 1)

    print("Fitting the learning data")
    clf.fit(X, Y)

    # test data

    # normalize data
    TESTpreIctalPSD = np.divide(TESTpreIctalPSD.reshape(-1, 1), MaxValue)
    TESTictalPSD = np.divide(TESTictalPSD.reshape(-1, 1), MaxValue)

    y = clf.predict(TESTpreIctalPSD)
    y1 = clf.predict(TESTictalPSD)

    y.reshape(-1, 1)
    y1.reshape(-1, 1)

    correct = np.where(y == [PreICTAL])

    correct1 = np.where(y1 == [ICTAL])


    importData[ratio[0]][ratio[1]]['ID' + str(number)]['preIctal'] \
        = correct[0].size / y.size

    importData[ratio[0]][ratio[1]]['ID' + str(number)]['Ictal'] = \
        correct1[0].size / y1.size

    #write the ratios
    '''print('Preictal accuracy: ' + str(correct[0].size/y.size))

    print('Preictal + PostIctal Ictal accuracy: '+
          str((correct[0].size+partlycorrect[0].size)/y.size))

    print('Ictal accuracy: ' + str(correct1[0].size/y1.size))

    print('Postictal accuracy: ' + str(correct2[0].size/y2.size))

    print('PostIctal + PreIctal accuracy: ' +
          str((correct2[0].size + partlycorrect2[0].size) / y2.size))'''

    return importData
def writeToFile(file, dict):
    '''Writes data to a json file for later use'''
    with open(file, "w") as f:
        json.dump(dict, f)
    f.close()
    return
def readFile(file):
    '''Reads a json file info
    :return dict (of ratios)'''
    with open(file, 'r') as f:
        dict = json.load(f)
    print('dictionary read successfully from file')
    f.close()
    # change the preIctal,Ictal and postIctal values back to int for easier
    # processing
    dict['learning'] = {int(k): v for k, v in dict['learning'].items()}
    dict['test'] = {int(k): v for k, v in dict['test'].items()}

    return dict
def butterword_filter(data, f, filtertype, order=4):
    '''Butterword filter. used for sub-band division if wanted for
    more spesific inspection
    :return ndarray (filtered signal)'''
    if filtertype == 'lowpass' or filtertype == 'highpass':
        b, a = signal.butter(order, f[1], btype=filtertype, fs=fs)
        y = signal.filtfilt(b, a, data)
        return y
    else:
        wn = f
        b, a = signal.butter(order, wn, btype=filtertype, fs=fs)
        y = signal.filtfilt(b, a, data)
        return y
def nfilter(data, r_f, fs, quality_factor=20):
    '''Notch filter from signal library.
     Can be used for line current oscillation removal.
     :return ndarray(filtered signal y)'''
    b, a = signal.iirnotch(r_f, quality_factor, fs)
    y = signal.filtfilt(b, a, data)
    return y

    return y
def loadData(filename):
    '''Loads mat datafile and takes values form the EEG signal founded
    :return dict (EEG) '''
    EEG = sio.loadmat(filename)
    EEG = EEG['EEG']
    return EEG
def basicPlot(EEG, freq,time):
    '''Draws plot of a EEG signal with the indicators of ictal phase'''
    plt.plot(freq, EEG, lw=1.5, color='k')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EEG Data in the Time Domain')

    plt.show()
def powerSpectruDistrubutionPlot(pre, ictal, post, time):
    '''Used to Visualize PSD spectrum of a signal'''
    for EEGData in [pre, ictal, post]:
        frequencies, power_spectrum = signal.welch(EEGData, fs=fs)
        freq = np.arange(0, time, 1 / fs)
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, 10 * np.log10(power_spectrum))
        plt.ylabel('Power dB(V^2/Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.xlim([0, 150])
        plt.title('EEG Power Spectral Density')
        plt.tight_layout()
        plt.show()
def getPower(freq, EEG):
    '''Gets signals PSD with welch method. Divides the PSD to sub-bands. Gets
    the mean of the PSD
    :return list, (of int values representing PSD sub-bands means)'''
    freqs, PSD = signal.welch(EEG, fs, nperseg=(1024))

    '''plt.semilogy(freqs, PSD, color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.title("Welch's periodogram")
        plt.xlim([0, 150])
        plt.show()'''

    # default is Hann window
    dividedPSD = []
    for band in [delta_band, theta_band, alpha_band, beta_band,
                 low_gamma_band, high_gamma_band]:
        indexMIN = np.where(freqs == band[0])[0][0]
        indexMAX = np.where(freqs == band[1])[0][0]
        dividedPSD.append(np.mean(PSD[indexMIN:indexMAX]))

    return dividedPSD
def toBands(EEG, freq, time):
    '''Divides the signal to sub-bands, plots them,
    :return ndarray, sub-band signals with the fixed length
    of the original signal'''
    ##divide the signals to the frequency bands
    raw = EEG

    delta = butterword_filter(EEG, delta_band, 'lowpass')

    theta = butterword_filter(EEG, theta_band, 'bandpass')

    alpha = butterword_filter(EEG, alpha_band, 'bandpass')

    beta = butterword_filter(EEG, beta_band, 'bandpass')

    low_gamma = butterword_filter(EEG, low_gamma_band, 'bandpass')

    high_gamma = butterword_filter(EEG, high_gamma_band, 'bandpass')

    # Plot EEG data in frequency bands
    plt.figure(figsize=(11, 9))
    plt.ylabel('Amplitude (µV)',labelpad = 50)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(7, 1, 1)
    plt.plot(freq, delta, lw=1.5)
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('Filtered EEG')
    plt.xticks([])

    plt.subplot(7, 1, 2)
    plt.plot(freq, delta, lw=1.5, color='b')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('Delta Band 0,5 - 4 Hz')
    plt.xticks([])

    plt.subplot(7, 1, 3)
    plt.plot(freq, theta, lw=1.5, color='g')
    plt.axvline(180, color='r')  # seizure start
    plt.axvline(time - 180, color='r')  # seizure end
    plt.title('Theta Band 4 - 8 Hz ')
    plt.xticks([])

    plt.subplot(7, 1, 4)
    plt.plot(freq, alpha, lw=1.5, color='r')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('Alpha Band 8 - 13 Hz')
    plt.xticks([])

    plt.subplot(7, 1, 5)
    plt.plot(freq, beta, lw=1.5, color='c')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('Beta Band 13 - 30 Hz')
    plt.xticks([])

    plt.subplot(7, 1, 6)
    plt.plot(freq, low_gamma, lw=1.5, color='m')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('Low_Gamma Band 30 - 80 Hz')
    plt.xticks([])

    plt.subplot(7, 1, 7)
    plt.plot(freq, high_gamma, lw=1.5, color='y')
    plt.axvline(180, color='g')  # seizure start
    plt.axvline(time - 180, color='g')  # seizure end
    plt.title('High_Gamma Band 80 - 145 Hz')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    return delta, theta, alpha, beta, low_gamma, high_gamma
if __name__ == '__main__':
    main()
