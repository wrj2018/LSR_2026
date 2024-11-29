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
parser.add_argument("-new_seed", "--new_seed", default=24, type=int)
args = parser.parse_args()
np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)

direct_viz_samples = False

data_folder = "12_ML_data"; data_dict = {}
decoded_samples = np.load("decoded_samples.npy")
mat_keys = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]

y_labels = [r'$C_{11}$',                  # 'C11'
            r'$C_{12}$',                  # 'C12'
            r'$C_{44}$',                  # 'C44'
            r'$E_{\rm coh}$',             # 'Cohesive_energy'
            r'$\tau^{110}_{\rm iss}$',    # 'ISS_110'
            r'$\tau^{112}_{\rm iss}$',    # 'ISS_112'
            r'$\tau^{123}_{\rm iss}$',    # 'ISS_123'
            r'$a_0$',                     # 'Lattice_constants'
            r'$\delta$',                  # 'Normalized_LD'
            r'$\gamma^{110}_{\rm usf}$',  # 'USFE_110'
            r'$\gamma^{112}_{\rm usf}$',  # 'USFE_112'
            r'$\gamma^{123}_{\rm usf}$'   # 'USFE_123'
            ]

x_labels = [r"$\rm edge\ \{110\}$",   # 'LSR_edge_110'
            r"$\rm edge\  \{112\}$",  # 'LSR_edge_112'
            r"$\rm edge\  \{123\}$",  # 'LSR_edge_123'
            r"$\rm screw\ \{110\}$",  # 'LSR_screw_110'
            r"$\rm screw\  \{112\}$", # 'LSR_screw_112'
            r"$\rm screw\  \{123\}$"  # 'LSR_screw_123'
            ]

spearman_all_mat = np.zeros((6, 12))#, np.zeros((6, 6))

for material in tqdm(range(12)): # evaluating the material parameters
    matl_spearman_coeff_mat, matl_pval_coeff_mat = np.zeros((6, 12)), np.zeros((6, 12))
    # dirc_spearman_coeff_mat, dirc_pval_coeff_mat = np.zeros((6, 6)), np.zeros((6, 6))

    stress_dat = decoded_samples[material, :, :6]; material_dat = decoded_samples[material, :, :]
    
    if direct_viz_samples:
        fig, axes = plt.subplots(6, 12, figsize=(30, 15))  # Create a 6x6 grid of subplots
        fig.suptitle(f'Correlation for Material {mat_keys[material]}', fontsize=16)

    for i in range(6): # stress data -> 112, 123, 110
        for j in range(12): # material data
            min_len = 1000
            
            strs_dat = stress_dat[:,i]
            material_lbl_dat = material_dat #[:,mat_label] # REORDER
            # direction_lbl_dat = material_dat #[:,dir_label] # REORDER

            matl_dat = material_lbl_dat[:,j]; #dirc_dat = direction_lbl_dat[:,j]

            matl_spearman_coeff, matl_pval_coeff = stats.spearmanr(strs_dat, matl_dat)
            # dirc_spearman_coeff, dirc_pval_coeff = stats.spearmanr(strs_dat, dirc_dat)

            if matl_pval_coeff<=0.05: # only sample statistically significant event for material correlation
                matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = matl_spearman_coeff, matl_pval_coeff
            else: matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = 0, 0

            # if dirc_pval_coeff<=0.05: # only sample statistically significant event for directional correlation
            #     dirc_spearman_coeff_mat[i, j], dirc_pval_coeff_mat[i, j] = dirc_spearman_coeff, dirc_pval_coeff
            # else: dirc_spearman_coeff_mat[i, j], dirc_pval_coeff_mat[i, j] = 0, 0
            
            if direct_viz_samples:
                ax = axes[i, j]; ax.scatter(strs_dat, matl_dat, alpha=0.7, s=5)
                ax.set_title(f"ρ={spearman_coeff:.2f}, κ={pval_coeff:.2f}", fontsize=10)  # Spearman coefficient as title
                ax.tick_params(axis='both', which='both', length=0, labelsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 5: ax.set_xlabel(f'{y_labels[j]}', fontsize=10)
                if j == 0: ax.set_ylabel(f'{x_labels[i]}', fontsize=10)
    
    if direct_viz_samples:
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f'fig/Material_{material}_correlation.png', dpi=300)  # Save the figure
        print(f'Figure saved! {mat_keys[material]}'); plt.close(fig)

    plot_coeff_mat(matl_spearman_coeff_mat, np.array(x_labels), np.array(y_labels), material, dir='fig', key='matl')
    # plot_coeff_mat(dirc_spearman_coeff_mat, np.array(x_labels), np.array(y_labels), material, dir='fig', key='dirc')

    spearman_all_mat += matl_spearman_coeff_mat
    # spearman_all_dir += dirc_spearman_coeff_mat

plot_coeff_mat(spearman_all_mat/12, np.array(x_labels), np.array(y_labels), material='all', dir='fig', key='matl_ALL')
# plot_coeff_mat(spearman_all_dir/12, np.array(x_labels), np.array(y_labels), material='all', dir='fig', key='dirc_ALL')

'''Interactions between stress components'''
stress_spearman_coeff_mat = np.zeros((6, 6))
stress_pval_coeff_mat = np.zeros((6, 6))

for material in tqdm(range(12)):
    for i in range(6):  # Stress component 1
        for j in range(6):  # Stress component 2
            if i != j:  # Skip self-correlation
                strs_dat_1 = stress_dat[:, i]; strs_dat_2 = stress_dat[:, j]

                stress_spearman_coeff, stress_pval_coeff = stats.spearmanr(strs_dat_1, strs_dat_2)

                if stress_pval_coeff <= 0.05:
                    tmp_stress_spearman_coeff_mat[i, j] = stress_spearman_coeff
                    stress_pval_coeff_mat[i, j] = stress_pval_coeff
                else:
                    tmp_stress_spearman_coeff_mat[i, j], stress_pval_coeff_mat[i, j] = 0, 0
    stress_spearman_coeff_mat += tmp_stress_spearman_coeff_mat

plot_coeff_mat(stress_spearman_coeff_mat/12, x_labels, x_labels, material, dir='fig', key='stress')