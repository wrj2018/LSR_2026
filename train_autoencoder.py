
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.sandbox.distributions.genpareto import shape
from tqdm import tqdm
from util import *
from sklearn.metrics import r2_score
import os, argparse, torch, random

parser = argparse.ArgumentParser(description="specify the parameters for the test case")
parser.add_argument("-seed", "--random_seed", default=152, type=int)
parser.add_argument("-repeat", "--repeat_samples", default=25, type=int)
parser.add_argument("-eig", "--eig_dim", default=3, type=int)
parser.add_argument("-hid", "--hid_dim", default=256, type=int)
parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
parser.add_argument("-ep", "--epochs", default=100000, type=int)
args = parser.parse_args()

os.makedirs('fig', exist_ok=True)
materials = ["1_NbTaTi", "2_MoNbTi", "3_HfNbTa", "4_NbTiZr", "5_HfNbTi", "6_HfTaTi", "7_TaTiZr", "8_MoTaTi", "9_MoNbTa", "10_HfNbTaTi", "11_HfMoNbTaTi", "12_HfNbTaTiZr"]
data_folder = "12_ML_data"; data_dict = {}

num_samples, mat_num, feat_num = 10, 12, 24 # number of samples in the raw data, total number of materials, raw feature space (stress + mat param.)
compressed_data = torch.zeros((mat_num, int(args.repeat_samples * num_samples), feat_num))
# norm_fact = torch.tensor([1,1,1,1,1,1,-100,10,100,100,100,10,10,1,1,1,100,10000])
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
                            10.0, # 'ISS_110'
                            10.0, # 'ISS_112'
                            10.0, # 'ISS_123'
                            10.0,   # 'Lattice_constants'
                            100.0, #Element Mo
                            100.0, #Element Nb
                            1000.0, # 'Normalized_LD'
                            100.0, #Element Ta
                            100.0, #Element Ti
                            1e-1,  # 'USFE_110'
                            1e-1,  # 'USFE_112'
                            1e-1,  # 'USFE_123'
                            100.0, # Element Zr
=======
                            10, #Mo
                            1e-1,  # 'USFE_112'
                            10.0, # 'ISS_112'
                            10.0, # 'ISS_110'
                            10, #Hf
                            10, #Ta
>>>>>>> d60e6c861123694d76dabc4309dbbffbcccf9bad
                            1e-1,   # 'LSR_edge_110'
                            1e-1,   # 'LSR_edge_112'
                            1e-1,   # 'LSR_edge_123'
                            1e-1, # 'LSR_screw_110'
                            1e-1, # 'LSR_screw_112'
                            1e-1, # 'LSR_screw_123'
                        ])

# ['12_HfNbTaTiZr_C11', '12_HfNbTaTiZr_C12', '12_HfNbTaTiZr_C44', 
# '12_HfNbTaTiZr_Cohesive_energy', '12_HfNbTaTiZr_Hf', 
# '12_HfNbTaTiZr_ISS_110', '12_HfNbTaTiZr_ISS_112', '12_HfNbTaTiZr_ISS_123', '12_HfNbTaTiZr_Lattice_constants', 
# '12_HfNbTaTiZr_Mo', '12_HfNbTaTiZr_Nb', 
# '12_HfNbTaTiZr_Normalized_LD', '12_HfNbTaTiZr_Ta', '12_HfNbTaTiZr_Ti', 
# '12_HfNbTaTiZr_USFE_110', '12_HfNbTaTiZr_USFE_112', '12_HfNbTaTiZr_USFE_123', '12_HfNbTaTiZr_Zr']

np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed); random.seed(args.random_seed)
seeds = np.random.randint(0, 9999, int(args.repeat_samples)).tolist()

for sample_id in tqdm(range(args.repeat_samples)):
    tmp_data = prepropress_data(data_folder, materials, norm_fact, num_samples, seeds[sample_id])
    compressed_data[:, sample_id*num_samples:(sample_id+1)*num_samples, :] = tmp_data


model = Autoencoder(eigen_dim=args.eig_dim, hidden_dim=args.hid_dim)

compressed_data_tensor = compressed_data.clone().detach().requires_grad_(True)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5) # scheduler for damping

loss_history = train_AutoEncoder(model, optimizer, loss_func, scheduler, input_data=compressed_data_tensor, epochs=args.epochs)

os.makedirs('model', exist_ok=True)
torch.save({'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()}, 
        f'model/MAT_AutoEnc_Eig3.pth')

with torch.no_grad():
    latent_dim_dat = model.encoder(compressed_data_tensor).numpy()
    reconstructed_data = model(compressed_data_tensor).numpy()
r2 = r2_score(compressed_data_tensor.flatten().detach().numpy(), reconstructed_data.flatten())

plot_pred(compressed_data_tensor, reconstructed_data, r2, file_name='autoencoder_reconstruct')
plot_loss(loss_history)
