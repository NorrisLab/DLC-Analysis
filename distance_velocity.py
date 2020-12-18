"""
Calculations for Cumulative Moving Distance and Velocity

Analysis performed using DeepLabCut

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


warnings.filterwarnings('ignore')

all_data = pd.DataFrame()
list_no = np.arange(0.0, 108060.0, 1.0) #number of frames in 30 minutes
# list_no = np.arange(0.0, 180000, 1.0) #number of frames in 50 minutes
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


    vel=vel*(1./fps)/100
    velocity_pd = pd.Series(vel)
    # vel_avg = velocity_pd.rolling(moving_average_duration_frames).mean()

    vel_avg = velocity_pd
    inst_dist = vel
    dist_sum = inst_dist.cumsum()

    # taking every 6th row
    every_6th = vel_avg
    nth_time = np.arange(len(every_6th))*.1
    every_6th_avg = every_6th
    new_df = every_6th_avg.copy()
    every_6th_avg_index = every_6th_avg.index

    all_data[video + "_dist"] = every_6th_avg.cumsum()[:108060]
    all_data[video + "_vel"] = new_df[:108060]
    time = np.arange(len(all_data[video + "_dist"]))

    all_data['time'] = time/60/60

if __name__ == '__main__':
    fig = plt.figure()
    """
    Saline
    """
    velocity(video='Saline_Ai14_OPRK1_C2_F0_Top Down', color='pink', label='F0 Saline')
    velocity(video='Saline_Ai14_OPRK1_C2_F1_Top Down', color='pink', label='F1 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_F2_Top Down', color='pink', label='F2 Saline')

    velocity(video='Saline_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M2_Top Down', color='pink', label='M2 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M3_Top Down', color='pink', label='M3 Saline')
    velocity(video='Saline_Ai14_OPRK1_C1_M4_Top Down', color='pink', label='M4 Saline')

    saline_avg_dist = all_data.loc[:,
               ['Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
                'Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_F2_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
                'Saline_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist Saline'] = saline_avg_dist.mean(axis=1)
    all_data['Avg Dist Saline SEM'] = stats.sem(saline_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist Saline'], color='black', linewidth=2,
             label='Saline+Saline')


    # saline_avg_vel = all_data.loc[:,
    #            ['Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Saline_Ai14_OPRK1_C1_M4_Top Down_vel']]
    # saline_avg_vel_noempty = saline_avg_vel.dropna()
    # avg_saline_vel_noempty = saline_avg_vel_noempty.mean(axis=1)
    # avg_saline_vel_noempty_sem = stats.sem(saline_avg_vel_noempty, axis=1)
    #
    # index_vals_sal = saline_avg_vel_noempty.index
    # better_time_sal = index_vals_sal * (1. / fps) / 60
    # vel_only_df_sal = avg_saline_vel_noempty
    # vel_only_df_sal_sem = avg_saline_vel_noempty_sem
    # # plt.plot(better_time_sal, vel_only_df_sal, color='black', linewidth=1,
    # #          label='Average Velocity Saline')
    #
    # better_time_sal_df = pd.DataFrame(data=better_time_sal)
    # index_vals_sal_df = pd.DataFrame(data=index_vals_sal)
    # saline_sum_index = saline_avg_vel_noempty.index
    # saline_avg_sum = saline_avg_vel_noempty*saline_sum_index
    # saline_avg_dist_mean = saline_avg_sum.mean(axis=1)
    # saline_avg_dist_sum = saline_avg_dist_mean.cumsum()
    # saline_avg_dist_sem = stats.sem(saline_avg_sum)
    # plt.plot(better_time_sal_df, saline_avg_dist_sum, color='black', linewidth=1, label='Average Cumulative Distance Saline')


    # #
    """
    U50
    """
    velocity(video='U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_F1_Top Down', color='orange', label='F1 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 Saline+5mgkg U50')

    velocity(video='U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 Saline+5mgkg U50')
    velocity(video='U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 Saline+5mgkg U50')

    u50_avg_dist = all_data.loc[:,
               ['U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'U50_Ai14_OPRK1_C1_F1_Top Down_dist',
                'U50_Ai14_OPRK1_C2_F2_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist U50'] = u50_avg_dist.mean(axis=1)
    all_data['Avg Dist U50 SEM'] = stats.sem(u50_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist U50'], color='#7ca338', linewidth=2,
             label='Saline+U50')
    #
    # u50_avg_vel = all_data.loc[:,
    #            ['U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_F1_Top Down_vel',
    #             'U50_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'U50_Ai14_OPRK1_C1_M4_Top Down_vel']]
    # # all_data['Avg Vel U50'] = u50_avg_vel.mean(axis=1)
    # # all_data['Avg Vel U50 SEM'] = stats.sem(u50_avg_vel, axis=1)
    #
    # # plt.plot(all_data['time'], all_data['Avg Vel U50'], color='orange', linewidth=1,
    # #          label='Average Vel U50')
    #
    # # vel_only_df_u50 = u50_avg_vel.mean(axis=1)
    # # vel_only_df_u50 = vel_only_df_u50.dropna()
    # # vel_only_df_u50_sem = stats.sem(u50_avg_vel, axis=1)
    # # index_vals_u50 =  vel_only_df_u50.index
    # # better_time_u50 = index_vals_u50 * (1. / fps) / 60
    # # plt.plot(better_time_u50, vel_only_df_u50, color='orange', linewidth=1,
    # #          label='Average Velocity U50')
    #
    # u50_avg_vel_noempty = u50_avg_vel.dropna()
    # avg_u50_vel_noempty = u50_avg_vel_noempty.mean(axis=1)
    # avg_u50_vel_noempty_sem = stats.sem(u50_avg_vel_noempty, axis=1)
    # vel_only_df_u50_sem = avg_u50_vel_noempty_sem
    #
    # index_vals_u50 = avg_u50_vel_noempty.index
    # better_time_u50 = index_vals_u50 * (1. / fps) / 60
    # vel_only_df_u50 = avg_u50_vel_noempty
    # plt.plot(better_time_u50, vel_only_df_u50, color='orange', linewidth=1,
    #          label='Average Velocity U50')

    """
    Naltrexone U50
    """
    velocity(video='Naltr_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_F1_Top Down', color='orange', label='F1 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_F2_Top Down', color='purple', label='F2 3mgkg Nalt+5mgkg U50')

    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 3mgkg Nalt+5mgkg U50')
    velocity(video='Nalt_U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 3mgkg Nalt+5mgkg U50')

    nalt_avg_dist = all_data.loc[:,
               ['Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist Nalt'] = nalt_avg_dist.mean(axis=1)
    all_data['Avg Dist Nalt SEM'] = stats.sem(nalt_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist Nalt'], color='#FF2500', linewidth=2,
             label='Naltrexone+U50')
    #
    # nalt_avg_vel = all_data.loc[:,
    #            [
    #             'Naltr_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_F2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'Nalt_U50_Ai14_OPRK1_C1_M4_Top Down_vel'
    #             ]]
    #
    #
    # all_data['Avg Vel Nalt'] = nalt_avg_vel.mean(axis=1)
    # all_data['Avg Vel Nalt SEM'] = stats.sem(nalt_avg_vel, axis=1)
    #
    # # plt.plot(all_data['time'], all_data['Avg Vel Nalt'], color='red', linewidth=1,
    # #          label='Average Velocity Nalt')
    # #
    # # # vel_only_df_nal = nalt_avg_vel.mean(axis=1)
    # # # vel_only_df_nal = vel_only_df_nal.dropna()
    # # # vel_only_df_nal_sem = stats.sem(nalt_avg_vel, axis=1)
    # # # index_vals_nal =  vel_only_df_nal.index
    # # # better_time_nal = index_vals_nal * (1. / fps) / 60
    # # # plt.plot(better_time_nal, vel_only_df_nal, color='red', linewidth=1,
    # # #          label='Average Velocity Nalt')
    # #
    # # nalt_avg_vel_noempty = nalt_avg_vel.dropna()
    # # avg_nalt_vel_noempty = nalt_avg_vel_noempty.mean(axis=1)
    # # avg_nalt_vel_noempty_sem = stats.sem(nalt_avg_vel_noempty, axis=1)
    # #
    # # index_vals_nal = nalt_avg_vel_noempty.index
    # # better_time_nal = index_vals_nal * (1. / fps) / 60
    # # vel_only_df_nal = avg_nalt_vel_noempty
    # # vel_only_df_nal_sem = avg_saline_vel_noempty_sem
    # # plt.plot(better_time_nal, vel_only_df_nal, color='#FF2500', linewidth=1,
    # #          label='Average Velocity Nalt')
    # #
    """
    NORBNI U50
    """
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down', color='orange', label='F1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down', color='purple', label='F2 10mgkg NORBNI+5mgkg U50')

    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down', color='pink', label='M1 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down', color='black', label='M2 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down', color='green', label='M3 10mgkg NORBNI+5mgkg U50')
    velocity(video='NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down', color='silver', label='M4 10mgkg NORBNI+5mgkg U50')
    norbni_u50_avg_dist = all_data.loc[:,
               ['NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_dist',
                'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_dist']]
    all_data['Avg Dist NORBNI+U50'] = norbni_u50_avg_dist.mean(axis=1)
    all_data['Avg Dist NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist NORBNI+U50'], color='#3853A3', linewidth=2,
             label='NORBNI+U50')
    #
    # norbni_u50_avg_vel = all_data.loc[:,
    #            [
    #             'NORBNI_U50_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'NORBNI_U50_Ai14_OPRK1_C1_M4_Top Down_vel'
    #             ]]
    # all_data['Avg Vel NORBNI+U50'] = norbni_u50_avg_vel.mean(axis=1)
    # all_data['Avg Vel NORBNI+U50 SEM'] = stats.sem(norbni_u50_avg_vel, axis=1)
    #
    #
    # vel_only_df_nu = norbni_u50_avg_vel.mean(axis=1)
    # vel_only_df_nu = vel_only_df_nu.dropna()
    # norbni_u50_avg_vel.dropna()
    #
    # norbni_u50_avg_vel_noempty = norbni_u50_avg_vel.dropna()
    # avg_norbni_u50_noempty = norbni_u50_avg_vel_noempty.mean(axis=1)
    # avg_norbni_u50_noempty_sem = stats.sem(norbni_u50_avg_vel_noempty, axis=1)
    #
    # vel_only_df_nu_sem = avg_norbni_u50_noempty_sem
    # index_vals = avg_norbni_u50_noempty.index
    # better_time_nu = index_vals*(1./fps)/60
    # plt.plot(better_time_nu, vel_only_df_nu, color='#3853A3', linewidth=1,
    #          label='Average Velocity NORBNI+U50')
    # # vel_only_df_nu_sem = stats.sem(vel_only_df_nu)
    # # index_vals = vel_only_df_nu.index
    # # better_time_nu = index_vals*(1./fps)/60
    # # plt.plot(better_time_nu, vel_only_df_nu, color='blue', linewidth=1,
    # #          label='Average Velocity NORBNI+U50')
    #
    # # all_data['Avg Vel NORBNI+Saline'] = norbni_saline_avg_vel.mean(axis=1)
    # # plt.plot(all_data['time'], all_data['Avg Vel NORBNI+U50'], color='blue', linewidth=1,
    # #          label='Average Vel NORBNI+U50')
    #
    """
    # NORBNI + Saline
    # """
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down', color='red', label='F0 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down', color='blue', label='F1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down', color='green', label='F2 10mgkg NORBNI+Saline')

    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down', color='orange', label='M1 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down', color='brown', label='M2 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down', color='black', label='M3 10mgkg NORBNI+Saline')
    velocity(video='NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down', color='purple', label='M4 10mgkg NORBNI+Saline')
    #
    norbni_saline_avg_dist = all_data.loc[:,
               ['NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist',
                'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_dist']]
    all_data['Avg Dist NORBNI+Saline'] = norbni_saline_avg_dist.mean(axis=1)
    all_data['Avg Dist NORBNI+Saline SEM'] = stats.sem(norbni_saline_avg_dist, axis=1)

    plt.plot(all_data['time'], all_data['Avg Dist NORBNI+Saline'], color='#9d57ff', linewidth=2,
             label='NORBNI+Saline')
    # #
    # norbni_saline_avg_vel = all_data.loc[:,
    #            [
    #             'NORBNI_Saline_Ai14_OPRK1_C2_F0_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C2_F2_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M1_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M2_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M3_Top Down_vel',
    #             'NORBNI_Saline_Ai14_OPRK1_C1_M4_Top Down_vel'
    #            ]]
    # vel_only_df_og = norbni_saline_avg_vel.mean(axis=1)
    # vel_only_df = vel_only_df_og.dropna()
    #
    # norbni_saline_avg_vel_noempty = norbni_saline_avg_vel.dropna()
    # avg_norbni_saline_noempty = norbni_saline_avg_vel_noempty.mean(axis=1)
    # avg_norbni_saline_noempty_sem = stats.sem(norbni_saline_avg_vel_noempty, axis=1)
    #
    # norbni_saline_df = pd.DataFrame(data=vel_only_df)
    #
    #
    # vel_norbni_sal_sem = avg_norbni_saline_noempty_sem
    # index_vals_ns = norbni_saline_avg_vel_noempty.index
    # vel_only_df = avg_norbni_saline_noempty
    # better_time_ns = index_vals_ns*(1./fps)/60
    # plt.plot(better_time_ns, vel_only_df, color='purple', linewidth=1,
    #          label='Average Velocity NORBNI+Saline')

    plt.fill_between(all_data['time'], all_data["Avg Dist Saline"]-all_data["Avg Dist Saline SEM"],
                     all_data["Avg Dist Saline"]+all_data["Avg Dist Saline SEM"], alpha=0.15, facecolor='black')
    plt.fill_between(all_data['time'], all_data["Avg Dist Nalt"]-all_data["Avg Dist Nalt SEM"],
                     all_data["Avg Dist Nalt"]+all_data["Avg Dist Nalt SEM"], alpha=0.15, facecolor='#FF2500')
    plt.fill_between(all_data['time'], all_data["Avg Dist U50"]-all_data["Avg Dist U50 SEM"],
                     all_data["Avg Dist U50"]+all_data["Avg Dist U50 SEM"], alpha=0.15, facecolor='#7ca338')
    plt.fill_between(all_data['time'], all_data["Avg Dist NORBNI+U50"]-all_data["Avg Dist NORBNI+U50 SEM"],
                     all_data["Avg Dist NORBNI+U50"]+all_data["Avg Dist NORBNI+U50 SEM"], alpha=0.15, facecolor='#3853A3', edgecolor='#3853A3')
    plt.fill_between(all_data['time'], all_data["Avg Dist NORBNI+Saline"]-all_data["Avg Dist NORBNI+Saline SEM"],
                     all_data["Avg Dist NORBNI+Saline"]+all_data["Avg Dist NORBNI+Saline SEM"], alpha=0.15, facecolor='#9d57ff', edgecolor='#5f38a3')


    #
    # plt.fill_between(better_time_sal, vel_only_df_sal + vel_only_df_sal_sem,
    #                  vel_only_df_sal - vel_only_df_sal_sem, alpha=0.25, facecolor='black',
    #                  edgecolor='black')
    # plt.fill_between(better_time_u50, vel_only_df_u50 - vel_only_df_u50_sem,
    #                  vel_only_df_u50 + vel_only_df_u50_sem, alpha=0.25, facecolor='orange',
    #                  edgecolor='orange')
    # plt.fill_between(better_time_nal, vel_only_df_nal - vel_only_df_nal_sem,
    #                  vel_only_df_nal + vel_only_df_nal_sem, alpha=0.25, facecolor='#FF2500',
    #                  edgecolor='red')
    # plt.fill_between(better_time_nu, vel_only_df_nu - vel_only_df_nu_sem, vel_only_df_nu + vel_only_df_nu_sem, alpha=0.25, facecolor='#3853A3', edgecolor='#3853A3')
    # plt.fill_between(better_time_ns, vel_only_df - vel_norbni_sal_sem, vel_only_df + vel_norbni_sal_sem, alpha=0.25, facecolor='purple', edgecolor='purple')

    '-----------------------------------------------------------------------------------------------------------'
    """Graph formatting"""
    font = {'family': 'Arial',
            'size': 12}
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 1)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    # plt.xlabel('Time (minutes)', fontsize=12)
    # plt.ylabel('Instantaneous Velocity (cm/s)', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Instantaneous Velocity (cm/s) '+str(no_seconds)+' second average ', fontsize=12)
    # # plt.legend(loc='upper left')
    # leg = plt.legend(loc='upper right', fontsize=12, frameon=False)
    # for i in leg.legendHandles:
    #     i.set_linewidth(1)
    # plt.show()

    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Total Distance Travelled', fontsize=12)
    # plt.legend(loc='upper left')
    leg = plt.legend(loc='upper left', fontsize=12, frameon=False)
    for i in leg.legendHandles:
        i.set_linewidth(2)
    plt.show()


    fig.savefig("foo.pdf", bbox_inches='tight')
    all_data.to_csv("alldata.csv")
