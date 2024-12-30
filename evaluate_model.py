import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from util import *
from sklearn.metrics import r2_score
import os, argparse, torch, random

parser = argparse.ArgumentParser(description="specify the parameters for the test case")
parser.add_argument("-seed", "--random_seed", default=152, type=int)
parser.add_argument("-new_seed", "--new_seed", default=24, type=int)
args = parser.parse_args()
np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)

num_samples = 10; os.makedirs('fig', exist_ok=True)

data_folder = "12_ML_data"
materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi","7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
norm_fact = torch.tensor([  1.0,   # 'C11'
                            1.0,   # 'C12'
                            1.0,   # 'C44'
                            -10.0,   # 'Cohesive_energy'
                            10.0, # 'ISS_110'
                            10.0, # 'ISS_112'
                            10.0, # 'ISS_123'
                            10.0,   # 'Lattice_constants'
                            1000.0, # 'Normalized_LD'
                            1e-1,  # 'USFE_110'
                            1e-1,  # 'USFE_112'
                            1e-1,  # 'USFE_123'
                            1e-1,   # 'LSR_edge_110'
                            1e-1,   # 'LSR_edge_112'
                            1e-1,   # 'LSR_edge_123'
                            1e-1, # 'LSR_screw_110'
                            1e-1, # 'LSR_screw_112'
                            1e-1, # 'LSR_screw_123'
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0,
                            100.0
                        ])

compressed_data = prepropress_data(data_folder, materials, norm_fact, num_samples, args.new_seed)

model_path = 'model/MAT_AutoEnc_Eig3.pth'
model = Autoencoder(eigen_dim=3, hidden_dim=256)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
compressed_data_tensor = compressed_data.clone().detach().requires_grad_(False)

with torch.no_grad():
    reconstructed_data = model(compressed_data_tensor).numpy()
    latent_dim_dat = model.encoder(compressed_data_tensor).numpy()

r2 = r2_score(compressed_data_tensor.flatten().numpy(), reconstructed_data.flatten())
print(compressed_data_tensor.flatten().numpy().shape())
exit()
plot_pred(compressed_data_tensor, reconstructed_data, r2)