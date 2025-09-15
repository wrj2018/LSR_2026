#*********************************************************************************************************************
# This file trains the model to fit the strength-LSR relation
# Written by Wu-Rong Jian to process the data about LSR
# Usage: python3 Pytorch_fitting_strength_LSR.py
#*********************************************************************************************************************


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from util_fitting import *
################################################# Define model training settings #######################################################
torch.manual_seed(21)
np.random.seed(21)

# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################# Prepare data #######################################################
# Load data from files
YieldStress_file = "data/YieldStress4strength.txt"
GrainSize_file = "data/GrainSize4strength.txt"
StrainRate_file = "data/StrainRate4strength.txt"
Temp_file = "data/Temp4strength.txt"
LSR_edge_110_file = "data/edge110LSR4strength.txt"
LSR_edge_112_file = "data/edge112LSR4strength.txt"
LSR_edge_123_file = "data/edge123LSR4strength.txt"
LSR_screw_110_file = "data/screw110LSR4strength.txt"
LSR_screw_112_file = "data/screw112LSR4strength.txt"
LSR_screw_123_file = "data/screw123LSR4strength.txt"

YieldStress_exp_np = np.loadtxt(YieldStress_file, usecols=[1])
GrainSize_np = np.loadtxt(GrainSize_file, usecols=[1])
StrainRate_np = np.loadtxt(StrainRate_file, usecols=[1])
Temp_np = np.loadtxt(Temp_file, usecols=[1])

YieldStress_exp_tensor = torch.tensor(YieldStress_exp_np, dtype=torch.float64, device=device)
GrainSize_tensor = torch.tensor(GrainSize_np, dtype=torch.float64, device=device)
StrainRate_tensor = torch.tensor(StrainRate_np, dtype=torch.float64, device=device)
Temp_tensor = torch.tensor(Temp_np, dtype=torch.float64, device=device)

LSR_edge_110 = np.loadtxt(LSR_edge_110_file, usecols=[1])
LSR_edge_112 = np.loadtxt(LSR_edge_112_file, usecols=[1])
LSR_edge_123 = np.loadtxt(LSR_edge_123_file, usecols=[1])
LSR_screw_110 = np.loadtxt(LSR_screw_110_file, usecols=[1])
LSR_screw_112 = np.loadtxt(LSR_screw_112_file, usecols=[1])
LSR_screw_123 = np.loadtxt(LSR_screw_123_file, usecols=[1])
merged_LSR_np = np.column_stack((LSR_edge_110, LSR_screw_110, LSR_edge_112, LSR_screw_112, LSR_edge_123, LSR_screw_123))
LSR_tensor = torch.tensor(merged_LSR_np, dtype=torch.float64, device=device)

print("LSR_tensor shape:", LSR_tensor.shape)
print("Temp_tensor shape:", Temp_tensor.shape)
print("StrainRate_tensor:", StrainRate_tensor.shape)
print("GrainSize_tensor:", GrainSize_tensor.shape)


################################################# Train model #######################################################
# Create model, loss function, and optimizer
learning_rate = 0.001
num_epochs = 50000
OPTIMIZERS = {'SGD': optim.SGD, 'Adam': optim.Adam}
optimizer_name = 'Adam'

model = CustomModel().to(device)
criterion = nn.MSELoss()
optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr=learning_rate)

# List to store loss values for plotting
train_losses = []

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()
    outputs = model(LSR_tensor, Temp_tensor, StrainRate_tensor, GrainSize_tensor)
    loss = criterion(outputs, YieldStress_exp_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0 or epoch == 0:
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    train_losses.append(loss.item())

# Save the loss values to a file
with open('./outputs/loss.txt', 'w') as f:
    for value in train_losses:
        f.write(f"{value}\n")

if model.subModel1.fc.exponential:
    trained_exp_weights_1 = torch.exp(model.subModel1.fc.weight.data.clone())
else:
    trained_exp_weights_1 = model.subModel1.fc.weight.data.clone()

trained_exp_weights_1_np = trained_exp_weights_1.cpu().numpy()

trained_symmetric_weights_1 = model.subModel1.fc.get_symmetric_weights()

# Convert the tensor to numpy array
trained_symmetric_weights_1_np = trained_symmetric_weights_1.detach().cpu().numpy()

# Save interaction aij to a file
with open('./outputs/interaction_coefficients_aij.txt', 'w') as f:
    for value in trained_exp_weights_1_np:
        f.write(f"{value}\n")

# Save to CSV file
np.savetxt('./outputs/trained_symmetric_weights_1.csv', trained_symmetric_weights_1_np, delimiter=',', fmt='%.6f')

# Save param_deltaH to a file
with open('./outputs/param_deltaH.txt', 'w') as f:
    for row in model.param_deltaH.detach().cpu().numpy():
        f.write(' '.join(f"{value:.4f}" for value in row) + '\n')

# Save param_KHP to a file
with open('./outputs/param_KHP.txt', 'w') as f:
    for value in model.param_KHP.detach().cpu().numpy():
        f.write(f"{value}\n")

# Save the Model's state dictionary
torch.save(model.state_dict(), './outputs/Model.pth')

######### Model Prediction #########
# Initialize the model architecture
model = CustomModel().to(device)

# Load the trained model
model.load_state_dict(torch.load('./outputs/Model.pth'))

# Set the model to evaluation mode
model.eval()
with torch.no_grad():
    torch_YieldStress_prediction = model(LSR_tensor, Temp_tensor, StrainRate_tensor, GrainSize_tensor)

YieldStress_prediction_np = torch_YieldStress_prediction.cpu().numpy()

######### Data preparation for MPEAs #########
Ncount = [1, 2, 8, 7, 6, 9, 17]

YieldStress_exp_list = [None] * len(Ncount)
YieldStress_prediction_list = [None] * len(Ncount)

index_start = 0

for i in range(len(Ncount)):
    print(f'i = {i}')
    count = Ncount[i]
    print("count:", count)

    index_end = index_start + count

    YieldStress_exp_list[i] = YieldStress_exp_np[index_start:index_end]
    YieldStress_prediction_list[i] = YieldStress_prediction_np[index_start:index_end]

    index_start = index_end

################################################# Plot settings #######################################################
x_values = [0, 500, 1000, 1500, 2000, 2500, 3000]
y_values = [0, 500, 1000, 1500, 2000, 2500, 3000]

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
markers = ['o', '*', '^', 'v', '<', '>', 'D']
labels = ['NbTaTi', 'HfNbTa', 'NbTiZr', 'HfNbTi', 'HfTaTi', 'HfNbTaTi', 'HfNbTaTiZr']

################################################# Plot configuration performance ########################################
plt.figure(figsize=(10, 10))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28
plt.rcParams['mathtext.fontset'] = 'stix'

r2_configurations = [None] * len(Ncount)

for i in range(len(Ncount)):
    # Calculate R-squared
    r2 = r2_score(YieldStress_exp_list[i], YieldStress_prediction_list[i])
    r2_configurations[i] = r2

    plt.scatter(YieldStress_exp_list[i], YieldStress_prediction_list[i], color=colors[i], marker=markers[i], label=labels[i], s=50)

plt.plot(x_values, y_values, color='black', linestyle='--', linewidth=2.5)
plt.tick_params(axis='both', which='both', direction='in', width=2, length=6, top=True, right=True)
plt.xlabel('Experimental yield stress (MPa)')
plt.ylabel('Predicted yield stress (MPa)')
plt.xlim(0, 3000)
plt.ylim(0, 3000)
plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
plt.gca().xaxis.set_minor_locator(MultipleLocator(100))
plt.gca().yaxis.set_minor_locator(MultipleLocator(100))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Adjust this value for frame line thickness

plt.savefig('./figures/Comparison_yield_stress.png', bbox_inches='tight')
plt.savefig('./figures/Comparison_yield_stress.pdf', bbox_inches='tight')


# Save r2 scores to a file
with open('./outputs/r2_MPEAs.txt', 'w') as f:
    for value in r2_configurations:
        f.write(f"{value}\n")

################################################# Plot loss values #######################################################
txtfilename = './outputs/loss.txt'
train_loss = np.loadtxt(txtfilename)

plt.figure(figsize=(10, 10))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28
plt.rcParams['mathtext.fontset'] = 'stix'

plt.semilogy(train_loss, color=colors[0], label='training', linestyle='-', linewidth=2.5)
plt.tick_params(axis='both', which='both', direction='in')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right", frameon=False)
plt.savefig('./figures/loss.png', bbox_inches='tight')