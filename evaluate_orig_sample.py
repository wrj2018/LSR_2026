import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from util import *
import os, argparse

parser = argparse.ArgumentParser(description="specify the parameters for the test case")
parser.add_argument("-seed", "--random_seed", default=152, type=int)
parser.add_argument("-repeat", "--repeat_samples", default=25, type=int)
args = parser.parse_args()

data_folder = "12_ML_data"; data_dict = {}

num_samples = 10
compressed_data = torch.zeros((12, int(args.repeat_samples*num_samples), 24))
<<<<<<< HEAD
norm_fact = torch.tensor([  1.0,   # 'C11'
=======
norm_fact = torch.tensor([  1e-1,  # 'USFE_110'
                            1.0,   # 'C11'
                            1000, # 'Normalized_LD'
                            10.0,   # 'Lattice_constants'
                            10, #Zr
                            1e-1,  # 'USFE_123'
                            10.0, # 'ISS_123'
                            10, #Ti
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad
                            1.0,   # 'C12'
                            1.0,   # 'C44'
                            10, #Nb
                            -10.0,   # 'Cohesive_energy'
                            10, #Mo
                            1e-1,  # 'USFE_112'
                            10.0, # 'ISS_112'
                            10.0, # 'ISS_110'
                            10, #Hf
                            10, #Ta
                            1e-1,   # 'LSR_edge_110'
                            1e-1,   # 'LSR_edge_112'
                            1e-1,   # 'LSR_edge_123'
                            1e-1, # 'LSR_screw_110'
                            1e-1, # 'LSR_screw_112'
                            1e-1, # 'LSR_screw_123'1.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0
                        ])

np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)
seeds = np.random.randint(0, 9999, args.repeat_samples).tolist()
materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]

for sample_id in tqdm(range(args.repeat_samples)):
    tmp_data = prepropress_data(data_folder, materials, norm_fact, num_samples, seeds[sample_id])
    compressed_data[:, sample_id*num_samples:(sample_id+1)*num_samples, :] = tmp_data

mat_keys = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
y_labels = [r'$C_{11}$', r'$C_{12}$', r'$C_{44}$', r'$E_{\rm coh}$',\
    r'$\tau^{110}_{\rm iss}$', r'$\tau^{112}_{\rm iss}$', r'$\tau^{123}_{\rm iss}$',\
        r'$a_0$', r'$\delta$', r'$\gamma^{110}_{\rm usf}$', r'$\gamma^{112}_{\rm usf}$', r'$\gamma^{123}_{\rm usf}$', r'$Hf$', r'$Mo$', r'$Nb$', r'$Ta$', r'$Ti$', r'$Zr$' ]

x_labels = [r"$\rm edge\ \{110\}$", r"$\rm edge\  \{112\}$", r"$\rm edge\  \{123\}$", r"$\rm screw\ \{110\}$", r"$\rm screw\  \{112\}$", r"$\rm screw\  \{123\}$" ]
spearman_all_mat = np.zeros((6, 12))

for material in tqdm(range(12)): # evaluating the material parameters
    matl_spearman_coeff_mat, matl_pval_coeff_mat = np.zeros((6, 12)), np.zeros((6, 12))
    stress_dat = compressed_data[material, :, 12:]; material_dat = compressed_data[material, :, :12] # util: line->198

    fig, axes = plt.subplots(6, 12, figsize=(30, 15))  # Create a 6x6 grid of subplots
    fig.suptitle(f'Correlation for Material {mat_keys[material]}', fontsize=16)

    for i in range(6): # stress data -> 110, 112, 123
        for j in range(12): # material data
            min_len = 1000
            strs_dat = stress_dat[:,i]
            matl_dat = material_dat[:,j]

            matl_spearman_coeff, matl_pval_coeff = stats.spearmanr(strs_dat, matl_dat)

            if matl_pval_coeff<=0.05: # only sample statistically significant event for material correlation
                matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = matl_spearman_coeff, matl_pval_coeff
            else: matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = 0, 0

    plot_coeff_mat(matl_spearman_coeff_mat, np.array(x_labels), np.array(y_labels), material, dir='fig', key='matl_orig')
    spearman_all_mat += matl_spearman_coeff_mat
plot_coeff_mat(spearman_all_mat/12, np.array(x_labels), np.array(y_labels), material='all', dir='fig', key='matl_ALL_orig')