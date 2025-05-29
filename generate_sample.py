import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
import torch, tqdm
from util import *

parser = argparse.ArgumentParser(description="specify the parameters for the test case")
parser.add_argument("-seed", "--random_seed", default=152, type=int)
parser.add_argument("-eig", "--eig_dim", default=3, type=int)
parser.add_argument("-hid", "--hid_dim", default=256, type=int)
parser.add_argument("-samp", "--new_samples", default=1000, type=int)
args = parser.parse_args()

data_folder = "12_ML_data"
materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
norm_fact = torch.tensor([  1e-1,  # 'USFE_110'
                            1.0,   # 'C11'
                            1000, # 'Normalized_LD'
                            10.0,   # 'Lattice_constants'
                            10, #Zr
                            1e-1,  # 'USFE_123'
                            10.0, # 'ISS_123'
                            10, #Ti
                            1.0,   # 'C12'
                            1.0,   # 'C44'
                            10, #Nb
                            -10.0,   # 'Cohesive_energy'
<<<<<<< HEAD
                            100.0, #Element Hf
                            100.0, #Element Mo
                            100.0, #Element Nb
                            100.0, #Element TA
                            100.0, #Element Ti
                            100.0, #Element Zr
                            10.0, # 'ISS_110'
                            10.0, # 'ISS_112'
                            10.0, # 'ISS_123'
                            10.0,   # 'Lattice_constants'
                            1000.0, # 'Normalized_LD'
                            1e-1,  # 'USFE_110'
=======
                            10, #Mo
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad
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
                            1e-1, # 'LSR_screw_123'
                        ])
num_samples = 10

np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)

compressed_data = prepropress_data(data_folder, materials, norm_fact, num_samples, args.random_seed)
model_path = 'model/MAT_AutoEnc_Eig3.pth'
model = Autoencoder(eigen_dim=args.eig_dim, hidden_dim=args.hid_dim)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

compressed_data_tensor = compressed_data.clone().detach().requires_grad_(False)
with torch.no_grad():
    reconstructed_data = model(compressed_data_tensor).numpy()
    latent_dim_dat = model.encoder(compressed_data_tensor).numpy()
compressed_data = latent_dim_dat

def sample_convex_hull(points_3d, n_samples=1000):
    hull = ConvexHull(points_3d)
    hull_faces = hull.simplices

    sampled_points = []
    for _ in range(n_samples):
        face = hull_faces[np.random.randint(len(hull_faces))]
        vertices = points_3d[face]

        bary_coords = np.random.dirichlet([1, 1, 1])
        sampled_point = np.dot(bary_coords, vertices)
        sampled_points.append(sampled_point)

    return np.array(sampled_points)

def visualize_latent(points_3d):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.get_cmap('seismic', 12); norm = plt.Normalize(vmin=0, vmax=11)

    for mat_id in range(12):
        color = cmap(norm(mat_id))  # Map material ID to color
        ax.scatter(points_3d[mat_id, :, 0], points_3d[mat_id, :, 1], points_3d[mat_id, :, 2],
                   color=color, s=25, alpha=0.75)

    ax.set_title("Latent Space")
    ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$"); ax.set_zlabel(r"$\xi_3$")
    ax.set_box_aspect([1, 1, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label('MPEA ID', fontsize=12)
    cbar.set_ticks([])
    plt.tight_layout()
    plt.savefig('fig/latent_mat', transparent=True, dpi=300); plt.show()


def visualize_convex_hull_and_samples(points_3d, sampled_points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='blue', s=10, label='Original Points', alpha=0.5)
    hull = ConvexHull(points_3d); hull_faces = [points_3d[simplex] for simplex in hull.simplices]
    poly3d = Poly3DCollection(hull_faces, color='cyan', alpha=0.25, edgecolor='k')
    ax.add_collection3d(poly3d)

    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], color='red', s=10, label='Sampled Points', alpha=0.5)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Convex Hull for Latent Space", fontsize=15)
    ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$"); ax.set_zlabel(r"$\xi_3$")
    ax.legend(); plt.tight_layout()
    plt.savefig('fig/convex_hull_sample', transparent=True, dpi=300); plt.show()

assert compressed_data.shape[1] >= 3, "Data must have at least 3 dimensions for a 3D surface plot."
points_3d = compressed_data.reshape(120, 3)  # Extract the 3D latent space

n_samples = 12 * args.new_samples
sampled_points = sample_convex_hull(points_3d, n_samples=n_samples)

visualize_convex_hull_and_samples(points_3d, sampled_points)
visualize_latent(compressed_data)

sampled_points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
decoded_samples = model.decoder(sampled_points_tensor)
decoded_samples = torch.abs(decoded_samples.reshape(12,int(n_samples/12),24))

''' plot the histogram of original and reconstructed data '''
compressed_np, reconstructed_np = compressed_data_tensor.numpy(), reconstructed_data
cmap = cm.get_cmap('seismic', 12); norm = plt.Normalize(vmin=0, vmax=11)
fig, axes = plt.subplots(1, 2, figsize=(9, 5)); fig.tight_layout(pad=5)

for i in range(12):
    color = cmap(norm(i)) 
    stress_compressed = compressed_np[i, :, :6].flatten()  # Flatten to 1D
    stress_reconstructed = reconstructed_np[i, :, :6].flatten()
    params_compressed = compressed_np[i, :, 6:].flatten()
    params_reconstructed = reconstructed_np[i, :, 6:].flatten()

    ax = axes[0]
    ax.hist(stress_compressed, bins=10, alpha=0.5, color=color)
    ax.hist(params_compressed, bins=10, alpha=0.5, color=color)
    ax.set_yscale('log'); ax.grid()
    ax.set_xlabel('Original data', fontsize=15); ax.set_ylabel('Count', fontsize=15)

    ax = axes[1]
    ax.hist(stress_reconstructed, bins=10, alpha=0.5, color=color)
    ax.hist(params_reconstructed, bins=10, alpha=0.5, color=color)
    ax.set_yscale('log'); ax.grid()
    ax.set_xlabel('Reconstructed data', fontsize=15); ax.set_ylabel('Count', fontsize=15)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]); plt.tight_layout()

cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label('MPEA ID', fontsize=12)
cbar.set_ticks([])
plt.savefig('fig/fig_reconstr', dpi=300)
plt.close('all')

cmap = cm.get_cmap('seismic', 12); norm = plt.Normalize(vmin=0, vmax=11)
fig = plt.figure(figsize=(5, 5))
for i in range(12):
    color = cmap(norm(i)) 
    stress_reconstructed = decoded_samples[i, :, 19:24].flatten().detach().numpy()
    mat_reconstructed = decoded_samples[i, :, 0:18].flatten().detach().numpy()
    plt.hist(stress_reconstructed, bins=99, alpha=0.25, label=f'Material {i+1}', color=color)
    plt.hist(mat_reconstructed, bins=99, alpha=0.25, color=color)
    plt.yscale('log')
    plt.xlabel('Generated data', fontsize=15); plt.ylabel('Count', fontsize=15)

sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label('MPEA ID', fontsize=12)
cbar.set_ticks([]); plt.tight_layout()
plt.savefig('fig/fig_gen', dpi=300)

<<<<<<< HEAD
decoded_samples_np = decoded_samples.detach().cpu().numpy()
=======
decoded_samples_np = decoded_samples.detach().cpu().numpy()  # Convert to NumPy array
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad

#arjun
import pandas as pd
Alloys = ['NbTaTi', 'MoNbTi', 'HfNbTa', 'NbTiZr', 'HfNbTi', 'HfTaTi', 'TaTiZr',
          'MoTaTi', 'MoNbTa', 'HfNbTaTi', 'HfMoNbTaTi', 'HfNbTaTiZr']
<<<<<<< HEAD
mech_pro = ['1_NbTaTi_C11', '1_NbTaTi_C12', '1_NbTaTi_C44', '1_NbTaTi_Cohesive_energy', '1_NbTaTi_Hf', '1_NbTaTi_ISS_110', '1_NbTaTi_ISS_112', '1_NbTaTi_ISS_123', '1_NbTaTi_Lattice_constants', '1_NbTaTi_Mo', '1_NbTaTi_Nb', '1_NbTaTi_Normalized_LD', '1_NbTaTi_Ta', '1_NbTaTi_Ti', '1_NbTaTi_USFE_110', '1_NbTaTi_USFE_112', '1_NbTaTi_USFE_123', '1_NbTaTi_Zr']
stress_pars = ['1_NbTaTi_LSR_edge_110', '1_NbTaTi_LSR_edge_112', '1_NbTaTi_LSR_edge_123', '1_NbTaTi_LSR_screw_110', '1_NbTaTi_LSR_screw_112', '1_NbTaTi_LSR_screw_123']
=======
mech_pro = ['USFE_110', 'C11', 'Normalized_LD', 'Lattice_constants', 'Zr', 'USFE_123', 'ISS_123', 'Ti', 'C12', 'C44', 'Nb', 'Cohesive_energy', 'Mo', 'USFE_112', 'ISS_112', 'ISS_110', 'Hf', 'Ta']
stress_pars = ['LSR_edge_110', 'LSR_edge_112', 'LSR_edge_123', 'LSR_screw_110', 'LSR_screw_112', 'LSR_screw_123']
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad

features = ['Alloy'] + mech_pro + stress_pars
df = pd.DataFrame(columns=features)
for i in range(12):
    for j in range(1000):
        data = decoded_samples_np[i, j]
        data_list = data.tolist()
        data_list.insert(0, Alloys[i])
        #print(data_list)
        df.loc[len(df)] = data_list
df.to_csv('augmented_data.csv', index=False)

<<<<<<< HEAD
print(decoded_samples_np.shape)
=======
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad
print(f"Decoded samples shape: {decoded_samples.shape}, {decoded_samples_np.shape}")
np.save("decoded_samples.npy", np.abs(decoded_samples_np))  # Save as .npy file