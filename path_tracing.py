"""
Path Tracing for Figure

Analysis perfomed using DeepLabCut

@article{Mathisetal2018,
    title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal={Nature Neuroscience},
    year={2018},
    url={https://www.nature.com/articles/s41593-018-0209-y}}

 @article{NathMathisetal2019,
    title={Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
    author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
    journal={Nature Protocols},
    year={2019},
    url={https://doi.org/10.1038/s41596-019-0176-0}}

Based Code from Contributed by Federico Claudi (https://github.com/FedeClaudi) for DeepLabCut Utilities
"""

# Importing the toolbox (takes several seconds)
import warnings

import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import time_in_each_roi
from scipy import stats
from textwrap import wrap
from scipy import integrate
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

all_data = pd.DataFrame()
list_no = np.arange(0.0, 108060.0, 1.0) #number of frames in 30 minutes
ms_time = np.arange(0.0, 2670.0, 0.4) #1ms increments of time
# frame rate of camera in those experiments
start_frame = 120 #frame to start at
pick_frame = 30 # pick every __th frame

fps = 60
no_seconds = 30
moving_average_duration_frames = fps * no_seconds
updated_window = no_seconds/(pick_frame/fps)
updated_window = int(updated_window)

DLCscorer = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'

def velocity(video, color, label):

    dataname = str(Path(video).stem) + DLCscorer + '.h5'
    print(dataname)

    #loading output of DLC
    Dataframe = pd.read_hdf(os.path.join(dataname), errors='ignore')
    # Dataframe.reset_index(drop=True)

    #you can read out the header to get body part names!
    bodyparts=Dataframe.columns.get_level_values(1)

    bodyparts2plot=bodyparts

    # let's calculate velocity of the back
    # this can be changed to whatever body part
    bpt='head'
    vel = time_in_each_roi.calc_distance_between_points_in_a_vector_2d(np.vstack([Dataframe[DLCscorer][bpt]['x'].values.flatten(), Dataframe[DLCscorer][bpt]['y'].values.flatten()]).T)
    x_y_cord_df = pd.DataFrame()
    x_y_cord_df['x'] = Dataframe[DLCscorer][bpt]['x'].values
    x_y_cord_df['y'] = Dataframe[DLCscorer][bpt]['y'].values
    print(x_y_cord_df)
    plt.plot(x_y_cord_df['x'], x_y_cord_df['y'], color=color, label=label)


if __name__ == '__main__':
    fig = plt.figure()

    """
    U50
    """
    velocity(video='U50_Ai14_OPRK1_C1_F1_Top Down', color='#7ca338', label='F1 Saline+5mgkg U50')

    '-----------------------------------------------------------------------------------------------------------'
    """Graph formatting"""
    font = {'family': 'Arial',
            'size': 12}
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 1)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title('Path travelled U50', fontsize=12)
    plt.show()

    # pp = PdfPages("foo.pdf")
    # pp.savefig(fig, bbox_inches='tight')
    # pp.close()
    # fig.savefig("foo.pdf", bbox_inches='tight')
    plt.savefig('U50_Path.eps', format='eps')
