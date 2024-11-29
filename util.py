import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os, argparse, torch, random


def l2_norm(mat1, mat2):
    return np.linalg.norm(mat1 - mat2, 'fro')


def plot_error_convergence(num_seeds, spearman_matrices, final_spearman_coeff_mat, material='1_NbTaTi', dir=None):
    errors = []
    for n in range(1, num_seeds + 1):
        avg_spearman_coeff_mat = np.mean(spearman_matrices[:n], axis=0)
        error = l2_norm(avg_spearman_coeff_mat, final_spearman_coeff_mat)
        errors.append(error)
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, num_seeds + 1), errors, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Seeds'); plt.ylabel('L2 Norm')
    plt.title('Decay of Error with Increasing Number of Seeds')
    plt.grid(True)
    plt.savefig(f'{dir}/l2_norm_{material}', dpi=300)
    del errors


# def plot_coeff_mat(final_spearman_coeff_mat, x_labels, y_labels, material='1_NbTaTi', dir=None, key=None):
#     if key=='dirc' or key=='dirc_ALL':
#         valid_ind = np.array([0,5],[1,5],[0,4],[1,4], [2,3],[3,3],[2,2],[3,2], [4,1],[5,1],[4,0],[5,0])
    # plt.figure(figsize=(5, 5))
    # plt.imshow(final_spearman_coeff_mat.T, cmap='seismic', vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.title(f'Spearman Correlation:\n {material}')
    # plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right')
    # plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

#     if key == 'dirc' or key == 'dirc_ALL':
#         for ind in valid_ind:
#             plt.scatter(ind[0], ind[1], color='red', s=100, marker='o', edgecolor='black', label='Valid Indices')
#         else:
#             for i in range(final_spearman_coeff_mat.shape[0]):
#                 for j in range(final_spearman_coeff_mat.shape[1]):
#                     if [i, j] not in valid_ind.tolist():  # Check if the index is not in valid_ind
#                         plt.scatter(i, j, color='black', s=50, marker='x', label='Invalid Indices')
#     plt.tight_layout()

#     plt.savefig(f"{dir}/spearman_coeff_{material}_{key}.png", dpi=300)
#     plt.close()


def plot_coeff_mat(final_spearman_coeff_mat, x_labels, y_labels, material='1_NbTaTi', dir=None, key=None, direct_corr=False):
    if key == 'dirc' or key == 'dirc_ALL':
        # valid_ind = np.array([[0, 5], [1, 5], [0, 4], [1, 4], [2, 3], [3, 3], [2, 2], [3, 2], [4, 1], [5, 1], [4, 0], [5, 0]])
        valid_ind = np.array([[0, 0],[1, 1],[0, 1],[1, 0], [2, 2],[3, 3],[2, 3],[3, 2], [4, 4],[5, 5],[4, 5],[5, 4]])

        plt.figure(figsize=(5, 5))
        plt.rcParams.update({
                "text.usetex": False, "font.family": "serif", "font.serif": "Times New Roman",
                "mathtext.fontset": "stix", "legend.fontsize": 10,
                "axes.labelsize": 12, "xtick.labelsize": 15, "ytick.labelsize": 15})
        if direct_corr:
            masked_matrix = np.ones_like(final_spearman_coeff_mat) * np.nan  # NaN for neutral
            masked_matrix[valid_ind[:, 0], valid_ind[:, 1]] = final_spearman_coeff_mat[valid_ind[:, 0], valid_ind[:, 1]]
            plt.imshow(masked_matrix.T, cmap='seismic', vmin=-1, vmax=1)

        else:
            plt.imshow(final_spearman_coeff_mat.T, cmap='seismic', vmin=-1, vmax=1)

        # plt.axvline(x=1.5, color='black', linewidth=1)  # Line at the 2nd column
        # plt.axvline(x=3.5, color='black', linewidth=1)  # Line at the 4th column

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10) 
        plt.title(r'Spearman Correlation', fontsize=15)
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

        for i in range(len(final_spearman_coeff_mat)):
            for j in range(len(final_spearman_coeff_mat[i])):
                if [i, j] not in valid_ind:
                    plt.scatter(i, j, color='black', s=100, marker='x', edgecolor='black')  # Mark invalid indices
       
        plt.tight_layout()
        plt.savefig(f"{dir}/spearman_coeff_{material}_{key}.png", dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(5, 5))
        plt.rcParams.update({
                "text.usetex": False, "font.family": "serif", "font.serif": "Times New Roman",
                "mathtext.fontset": "stix", "legend.fontsize": 10,
                "axes.labelsize": 12, "xtick.labelsize": 15, "ytick.labelsize": 15})
        plt.imshow(final_spearman_coeff_mat.T, cmap='seismic', vmin=-1, vmax=1)
        # plt.axvline(x=1.5, color='black', linewidth=1)  # Line at the 2nd column
        # plt.axvline(x=3.5, color='black', linewidth=1)  # Line at the 4th column
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10) 
        plt.title(r'Spearman Correlation', fontsize=15)
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
        plt.tight_layout()
        plt.savefig(f"{dir}/spearman_coeff_{material}_{key}.png", dpi=300)
        plt.close()


def train_AutoEncoder(model, optimizer, loss_func, scheduler, input_data=None, epochs=1000, prev_loss=None, \
    loss_history=[], stag_epochs=0):
    from tqdm import tqdm
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(input_data)
        loss = loss_func(output, input_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        if epoch % 1000 == 0:
            print(f'[{epoch}], Loss: {loss.item():.4f}')

        '''checking convergence'''
        if prev_loss is not None and loss == prev_loss: stag_epochs += 1
        else: stag_epochs = 0
        prev_loss = loss

        if stag_epochs >= 1000: # if len(loss_history)>=10000: #     if loss_history[-1]==loss_history[-1000]: 
            print('Autoencoder converged')
            break
    return loss_history


class Autoencoder(torch.nn.Module):
    def __init__(self, eigen_dim, hidden_dim=64):
        super(Autoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(18, hidden_dim),
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_dim, int(hidden_dim/2)), 
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim/2), eigen_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(eigen_dim, int(hidden_dim/2)), 
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim/2), hidden_dim),
            torch.nn.ReLU(), # 
            torch.nn.Linear(hidden_dim, 18)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def prepropress_data(data_folder, materials, norm_fact, num_samples=10, seed=None):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    data_dict = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path, delim_whitespace=True, header=None).values
                variable_name = os.path.splitext(file)[0]
                data_dict[variable_name] = data

    compressed_data = torch.zeros((len(materials), num_samples, len(norm_fact)))

    for i, material in enumerate(materials):
        material_data_dict = {key: data_dict[key] for key in data_dict if material in key}
        stress_data = {key: material_data_dict[key] for key in material_data_dict if "LSR" in key}
        material_data = {key: material_data_dict[key] for key in material_data_dict if "LSR" not in key}

        sorted_material_keys = sorted(material_data.keys())
        sorted_stress_keys = sorted(stress_data.keys())

        material_values = []
        for key in sorted_material_keys:
            # Convert stress data to tensors
            # stress_values = [torch.tensor(stress_data[key]) for key in stress_data]
            # for j in range(6):
            #     stress_values[j] = stress_values[j][torch.randperm(10)]
 
            # # Sample data for the current material
            # data_tensor = torch.tensor(material_data[key])
            # sampled_indices = random.sample(range(data_tensor.size(0)), num_samples)
            # sampled_values = data_tensor[sampled_indices]
            # material_values.append(sampled_values)
            stress_values = [torch.tensor(stress_data[k]) for k in sorted_stress_keys]
            data_tensor = torch.tensor(material_data[key])
            sampled_indices = random.sample(range(data_tensor.size(0)), num_samples)
            sampled_values = data_tensor[sampled_indices]
            material_values.append(sampled_values)
            

        stress_tensor = torch.stack(stress_values)
        materials_tensor = torch.stack(material_values)
        combined_tensor = torch.cat([materials_tensor, stress_tensor], dim=0).reshape(len(norm_fact), num_samples).T
        combined_tensor *= norm_fact
        compressed_data[i] = combined_tensor

    # compressed_data /= 100
    return compressed_data


def plot_pred(compressed_data_tensor, reconstructed_data, r2, file_name='autoencoder_reconstruction_evaluation'):
    plt.figure(figsize=(5, 5))
    plt.scatter(compressed_data_tensor.detach().numpy(), reconstructed_data, s=10, color='blue', alpha=0.25, label=rf'$R^2={r2:.2f}$')
    plt.plot([-10, 100], [-10, 100], '--', color='red')
    plt.ylabel(r'Reconstructed data', fontsize=15)
    plt.xlabel(r'Original data', fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'fig/{file_name}', dpi=300)
    plt.show()


def plot_loss(loss_history, file_name='loss_func'):
    plt.figure(figsize=(5, 5))
    plt.plot(loss_history, linewidth=3, alpha=0.75)
    plt.yscale('log')
    plt.xlabel(r'Epochs', fontsize=15); plt.ylabel(r'Loss', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'fig/{file_name}',dpi=300)