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
compressed_data = torch.zeros((12, int(args.repeat_samples*num_samples), 18))
norm_fact = torch.tensor([1,1,1,1,1,1,-100,10,100,100,100,10,10,1,1,1,100,10000])

np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)

seeds = np.random.randint(0, 9999, args.repeat_samples).tolist()
materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]

for sample_id in tqdm(range(args.repeat_samples)):
    tmp_data = prepropress_data(data_folder, materials, norm_fact, num_samples, seeds[sample_id])
    compressed_data[:, sample_id*num_samples:(sample_id+1)*num_samples, :] = tmp_data

mat_keys = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
y_labels = [r'$E_{\rm coh}$', r'$C_{12}$', r'$\tau^{112}_{\rm iss}$', r'$a_0$', r'$\tau^{123}_{\rm iss}$', r'$\gamma^{110}_{\rm usf}$', r'$C_{44}$', r'$C_{11}$', r'$\gamma^{112}_{\rm usf}$', r'$\gamma^{123}_{\rm usf}$', r'$\tau^{110}_{\rm iss}$', r'$\delta$']
x_labels = [r"$\rm screw\  \{112\}$", r"$\rm screw\  \{123\}$", r"$\rm edge\  \{123\}$", r"$\rm edge\  \{112\}$", r"$\rm edge\ \{110\}$", r"$\rm screw\  \{110\}$"]

spearman_all_mat, spearman_all_dir = np.zeros((6, 6)), np.zeros((6, 6))

for material in tqdm(range(12)): # evaluating the material parameters
    matl_spearman_coeff_mat, matl_pval_coeff_mat = np.zeros((6, 6)), np.zeros((6, 6))
    dirc_spearman_coeff_mat, dirc_pval_coeff_mat = np.zeros((6, 6)), np.zeros((6, 6))

    stress_dat = compressed_data[material, :, :6]; material_dat = compressed_data[material, :, 6:]

    fig, axes = plt.subplots(6, 12, figsize=(30, 15))  # Create a 6x6 grid of subplots
    fig.suptitle(f'Correlation for Material {mat_keys[material]}', fontsize=16)

    for i in range(6): # stress data -> 112, 123, 110
        for j in range(6): # material data
            min_len = 1000; # mat_label = [0, 4, 8, 9, 10, 11]; dir_label = [2, 7, 3, 6, 1, 5]
            mat_label = [0,1,3,6,7,11]; dir_label = [2,8,4,9,5,10] # ISS, USFE -> 112, 123, 110
            X_label = [0,3,1,2,4,5]

            stress_dat = stress_dat[:,X_label] # REORDER
            strs_dat = stress_dat[:,i]

            material_lbl_dat = material_dat[:,mat_label] # REORDER
            direction_lbl_dat = material_dat[:,dir_label] # REORDER

            matl_dat = material_lbl_dat[:,j]; dirc_dat = direction_lbl_dat[:,j]

            matl_spearman_coeff, matl_pval_coeff = stats.spearmanr(strs_dat, matl_dat)
            dirc_spearman_coeff, dirc_pval_coeff = stats.spearmanr(strs_dat, dirc_dat)

            if matl_pval_coeff<=0.05: # only sample statistically significant event for material correlation
                matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = matl_spearman_coeff, matl_pval_coeff
            else: matl_spearman_coeff_mat[i, j], matl_pval_coeff_mat[i, j] = 0, 0

            if dirc_pval_coeff<=0.05: # only sample statistically significant event for directional correlation
                dirc_spearman_coeff_mat[i, j], dirc_pval_coeff_mat[i, j] = dirc_spearman_coeff, dirc_pval_coeff
            else: dirc_spearman_coeff_mat[i, j], dirc_pval_coeff_mat[i, j] = 0, 0

    plot_coeff_mat(matl_spearman_coeff_mat, np.array(x_labels)[X_label], np.array(y_labels)[mat_label], material, dir='fig', key='matl_orig')
    plot_coeff_mat(dirc_spearman_coeff_mat, np.array(x_labels)[X_label], np.array(y_labels)[dir_label], material, dir='fig', key='dirc_orig')

    spearman_all_mat += matl_spearman_coeff_mat
    spearman_all_dir += dirc_spearman_coeff_mat

plot_coeff_mat(spearman_all_mat/12, np.array(x_labels)[X_label], np.array(y_labels)[mat_label], material='all', dir='fig', key='matl_ALL_orig')
plot_coeff_mat(spearman_all_dir/12, np.array(x_labels)[X_label], np.array(y_labels)[dir_label], material='all', dir='fig', key='dirc_ALL_orig')