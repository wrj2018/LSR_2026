import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from util import *

parser = argparse.ArgumentParser(description="specify the parameters for the test case")
parser.add_argument("-itr", "--seeds_iterations", default=100, type=int)
args = parser.parse_args()

num_seeds = args.seeds_iterations
data_folder = "12_ML_data"; data_dict = {}

for root, _, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)

            data = pd.read_csv(file_path, delim_whitespace=True, header=None).values
            variable_name = os.path.splitext(file)[0]
            data_dict[variable_name] = data

all_keys = data_dict.keys()

materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
x_labels = ["LSR_edge_112", "LSR_screw_112", "LSR_edge_123", "LSR_screw_123", "LSR_edge_110", "LSR_screw_110"]
y_labels = ["Normalized_LD", "ISS_110", "ISS_112", "ISS_123", "Cohesive_energy", "USFE_110", "USFE_123", "USFE_112", "C44", "C12", "C11", "Lattice_constants"]

os.makedirs('fig', exist_ok=True)
for material in materials:
    material_data_dict = {key: data_dict[key] for key in data_dict if material in key}

    stress_data = {key: material_data_dict[key] for key in material_data_dict if "LSR" in key}
    material_data = {key: material_data_dict[key] for key in material_data_dict if "LSR" not in key}

    stress_keys = list(stress_data.keys()); material_keys = list(material_data.keys())

    spearman_matrices = np.zeros((num_seeds, len(stress_keys), len(material_keys)))
    pval_matrices = np.zeros((num_seeds, len(stress_keys), len(material_keys)))

    for seed in tqdm(range(num_seeds), desc=f"Processing {material}"):
        np.random.seed(112 + seed)
        spearman_coeff_mat = np.zeros((len(stress_keys), len(material_keys)))
        pval_coeff_mat = np.zeros((len(stress_keys), len(material_keys)))

        for i, stress_key in enumerate(stress_keys):
            for j, material_key in enumerate(material_keys):
                min_len = min(len(stress_data[stress_key]), len(material_data[material_key])); valid = False

                while not valid:
                    strss_ind = np.random.choice(len(stress_data[stress_key]), min_len, replace=False)
                    matrl_ind = np.random.choice(len(material_data[material_key]), min_len, replace=False)

                    stress_dat = stress_data[stress_key][strss_ind]; material_dat = material_data[material_key][matrl_ind]
                    spearman_coeff, pval_coeff = stats.spearmanr(stress_dat, material_dat)

                    if pval_coeff < 0.05: # only sample the statistically significant event
                        spearman_coeff_mat[i, j] = spearman_coeff
                        pval_coeff_mat[i, j] = pval_coeff
                        valid = True

        spearman_matrices[seed] = spearman_coeff_mat; pval_matrices[seed] = pval_coeff_mat

    final_spearman_coeff_mat = np.mean(spearman_matrices, axis=0); final_pval_coeff_mat = np.mean(pval_matrices, axis=0)
    plot_error_convergence(num_seeds, spearman_matrices, final_spearman_coeff_mat, material, dir='fig')
    plot_coeff_mat(final_spearman_coeff_mat, x_labels, y_labels, material, dir='fig')