#*********************************************************************************************************************
# This file trains the model to fit the strength-LSR relation
# Written by Wu-Rong Jian to process the data about LSR
# Usage: python3 Pytorch_fitting_strength_LSR.py
#*********************************************************************************************************************

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

################################################# Define model training settings #######################################################
torch.manual_seed(21)
np.random.seed(21)

# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Functions #
def f1(LSR_input, Temp_input, Srate_input, deltaH_input):
    print("f1_LSR_input shape:", LSR_input.shape)
    print("f1_Temp_input shape:", Temp_input.shape)
    print("f1_Srate_input shape:", Srate_input.shape)
    print("f1_deltaH_input shape:", deltaH_input.shape)

    Temp_input_expanded = Temp_input.unsqueeze(1).expand(-1, 6)  # Shape becomes [50, 6]
    Srate_input_expanded = Srate_input.unsqueeze(1).expand(-1, 6)  # Shape becomes [50, 6]

    # Boltzmann_constant; Unit: eV/K
    Kb = torch.tensor([8.62e-5], dtype=torch.float64, device=device)
    # Reference strain rate; Unit: s^{-1}
    epsilon0_dot = torch.tensor([10000], dtype=torch.float64, device=device)
    output_tensor = LSR_input*(1-torch.pow((Kb*Temp_input_expanded*torch.log(epsilon0_dot/Srate_input_expanded)/deltaH_input), 2/3))
    return output_tensor

def f2(input_tensor):
    print("f2_input shape:", input_tensor.shape)
    output_tensor = torch.sum(input_tensor, dim=1)
    print("f2_output shape:", output_tensor.shape)
    return output_tensor

def f3(input_tensor, KHP_input, GrainSize_input):
    print("f3_input shape:", input_tensor.shape)
    print("f3_KHP_input shape:", KHP_input.shape)
    print("f3_GrainSize_input shape:", GrainSize_input.shape)
    
    # Taylor factor for BCC metals with random texture
    param_M = torch.tensor([2.733], dtype=torch.float64, device=device)
    YieldStress = input_tensor*param_M+KHP_input*torch.pow(GrainSize_input, -0.5)
    return YieldStress

# Symmetric linear layer definition #
class symmetric_linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, exponential=False):
        super(symmetric_linear, self).__init__()
        
        # Ensure in_features equals out_features for main diagonal symmetry
        assert in_features == out_features == 6, "in_features and out_features both should be 6"
        
        # Only store the upper triangular weights (including the diagonal)
        self.weight = nn.Parameter(torch.Tensor(21).double())
        
        if bias:
            # Initialize a single scalar value for bias
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)
        
        self.in_features = in_features
        self.out_features = out_features

        self.exponential = exponential  # This attribute will control the weight transformation
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights with a normal distribution, mean depending on `exponential`
        if self.exponential:
            nn.init.normal_(self.weight.data, mean=1, std=0.2)
        else:
            nn.init.normal_(self.weight.data, mean=1, std=0.1)

    def forward(self, input):
        symmetric_weight = self.get_symmetric_weights()

        # Apply the linear transformation
        output = input.matmul(symmetric_weight)
        if self.bias is not None:
            output += self.bias  # Broadcasting the bias

        return output

    def get_symmetric_weights(self):
        # Once your model has been trained, you can retrieve the trained symmetric weights by simply calling: 
        # trained_symmetric_weights = model.get_symmetric_weights()
        # Initialize the full symmetric weight matrix as zeros
        symmetric_weight = torch.zeros((self.in_features, self.out_features), device=device, dtype=torch.float64)

        # Conditional weight transformation based on the 'exponential' attribute
        if self.exponential:
            #exp_weight = 0.2 + torch.sigmoid(self.weight)
            exp_weight = 0.1 + torch.exp(self.weight)
        else:
            exp_weight = self.weight
    
        # Use the first weight (self) for the main diagonal
        # 0: edge 110, 1: screw 110, 2: edge 112, 3: screw 112, 4: edge 123, 5: screw 123
        symmetric_weight[0, 0] = exp_weight[0]
        symmetric_weight[0, 1] = exp_weight[1]
        symmetric_weight[0, 2] = exp_weight[2]
        symmetric_weight[0, 3] = exp_weight[3]
        symmetric_weight[0, 4] = exp_weight[4]
        symmetric_weight[0, 5] = exp_weight[5]
        symmetric_weight[1, 1] = exp_weight[6]
        symmetric_weight[1, 2] = exp_weight[7]
        symmetric_weight[1, 3] = exp_weight[8]
        symmetric_weight[1, 4] = exp_weight[9]
        symmetric_weight[1, 5] = exp_weight[10]
        symmetric_weight[2, 2] = exp_weight[11]
        symmetric_weight[2, 3] = exp_weight[12]
        symmetric_weight[2, 4] = exp_weight[13]
        symmetric_weight[2, 5] = exp_weight[14]
        symmetric_weight[3, 3] = exp_weight[15]
        symmetric_weight[3, 4] = exp_weight[16]
        symmetric_weight[3, 5] = exp_weight[17]
        symmetric_weight[4, 4] = exp_weight[18]
        symmetric_weight[4, 5] = exp_weight[19]
        symmetric_weight[5, 5] = exp_weight[20]
    
        # Reflect the upper triangular weights to the lower triangular part to make it symmetric
        for i in range(0, 6):
            for j in range(i+1, 6):
                symmetric_weight[j, i] = symmetric_weight[i, j]

        return symmetric_weight

# SLP definition #
class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()

        # Processing tensor; Weights are positive; There is no bias
        self.fc = symmetric_linear(6, 6, bias=False, exponential=True)

    def forward(self, A):
        # Processing tensor A
        A = self.fc(A)

        return A

# Training model definition #
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Initialize the sub-model
        self.subModel1 = SLP()

        # Initialize a 7X6 tensor as a parameter for deltaH
        # 7 and 6 represent seven MPEA compositions and six types of LSR, respectively
        # each element is drawn from a normal distribution (0,1)
        self._raw_param_deltaH = nn.Parameter(torch.randn(7, 6))

        # Initialize a 7-element one-dimensional tensor as a parameter for KHP
        # 7 represents seven MPEA compositions
        self._raw_param_KHP = nn.Parameter(torch.randn(7))

    @property
    def param_deltaH(self):
        # Unit: eV, BCC range (0.3-3 eV)
        # Ensure the params are always within the range 0 and 10 using sigmoid function
        # return 5*torch.sigmoid(self._raw_param_deltaH) 
        # return torch.exp(self._raw_param_deltaH)
        return 0.1 + 4.9 * torch.sigmoid(self._raw_param_deltaH)


    @property
    def param_KHP(self):
        # Unit: MPa·µm⁻⁰·⁵, BCC range (200–800 MPa·µm⁻⁰·⁵)
        # Ensure the params are always positive using exponential function
        return torch.exp(self._raw_param_KHP)
        #return 200 + 600*torch.sigmoid(self._raw_param_KHP)

    def forward(self, LSR_input, Temp_input, Srate_input, GrainSize_input):
        # Define the matrix of param_deltaH
        array_deltaH = torch.zeros((50, 6))

        for i in range(6):
            array_deltaH[0, i] = self.param_deltaH[0, i]
            array_deltaH[1, i] = self.param_deltaH[1, i]
            array_deltaH[2, i] = self.param_deltaH[1, i]
            array_deltaH[3, i] = self.param_deltaH[2, i]
            array_deltaH[4, i] = self.param_deltaH[2, i]
            array_deltaH[5, i] = self.param_deltaH[2, i]
            array_deltaH[6, i] = self.param_deltaH[2, i]
            array_deltaH[7, i] = self.param_deltaH[2, i]
            array_deltaH[8, i] = self.param_deltaH[2, i]
            array_deltaH[9, i] = self.param_deltaH[2, i]
            array_deltaH[10, i] = self.param_deltaH[2, i]
            array_deltaH[11, i] = self.param_deltaH[3, i]
            array_deltaH[12, i] = self.param_deltaH[3, i]
            array_deltaH[13, i] = self.param_deltaH[3, i]
            array_deltaH[14, i] = self.param_deltaH[3, i]
            array_deltaH[15, i] = self.param_deltaH[3, i]
            array_deltaH[16, i] = self.param_deltaH[3, i]
            array_deltaH[17, i] = self.param_deltaH[3, i]
            array_deltaH[18, i] = self.param_deltaH[4, i]
            array_deltaH[19, i] = self.param_deltaH[4, i]
            array_deltaH[20, i] = self.param_deltaH[4, i]
            array_deltaH[21, i] = self.param_deltaH[4, i]
            array_deltaH[22, i] = self.param_deltaH[4, i]
            array_deltaH[23, i] = self.param_deltaH[4, i]
            array_deltaH[24, i] = self.param_deltaH[5, i]
            array_deltaH[25, i] = self.param_deltaH[5, i]
            array_deltaH[26, i] = self.param_deltaH[5, i]
            array_deltaH[27, i] = self.param_deltaH[5, i]
            array_deltaH[28, i] = self.param_deltaH[5, i]
            array_deltaH[29, i] = self.param_deltaH[5, i]
            array_deltaH[30, i] = self.param_deltaH[5, i]
            array_deltaH[31, i] = self.param_deltaH[5, i]
            array_deltaH[32, i] = self.param_deltaH[5, i]
            array_deltaH[33, i] = self.param_deltaH[6, i]
            array_deltaH[34, i] = self.param_deltaH[6, i]
            array_deltaH[35, i] = self.param_deltaH[6, i]
            array_deltaH[36, i] = self.param_deltaH[6, i]
            array_deltaH[37, i] = self.param_deltaH[6, i]
            array_deltaH[38, i] = self.param_deltaH[6, i]
            array_deltaH[39, i] = self.param_deltaH[6, i]
            array_deltaH[40, i] = self.param_deltaH[6, i]
            array_deltaH[41, i] = self.param_deltaH[6, i]
            array_deltaH[42, i] = self.param_deltaH[6, i]
            array_deltaH[43, i] = self.param_deltaH[6, i]
            array_deltaH[44, i] = self.param_deltaH[6, i]
            array_deltaH[45, i] = self.param_deltaH[6, i]
            array_deltaH[46, i] = self.param_deltaH[6, i]
            array_deltaH[47, i] = self.param_deltaH[6, i]
            array_deltaH[48, i] = self.param_deltaH[6, i]
            array_deltaH[49, i] = self.param_deltaH[6, i]

        # Define the array of param_KHP
        array_KHP = torch.zeros(50)

        array_KHP[0] = self.param_KHP[0]
        array_KHP[1] = self.param_KHP[1]
        array_KHP[2] = self.param_KHP[1]
        array_KHP[3] = self.param_KHP[2]
        array_KHP[4] = self.param_KHP[2]
        array_KHP[5] = self.param_KHP[2]
        array_KHP[6] = self.param_KHP[2]
        array_KHP[7] = self.param_KHP[2]
        array_KHP[8] = self.param_KHP[2]
        array_KHP[9] = self.param_KHP[2]
        array_KHP[10] = self.param_KHP[2]
        array_KHP[11] = self.param_KHP[3]
        array_KHP[12] = self.param_KHP[3]
        array_KHP[13] = self.param_KHP[3]
        array_KHP[14] = self.param_KHP[3]
        array_KHP[15] = self.param_KHP[3]
        array_KHP[16] = self.param_KHP[3]
        array_KHP[17] = self.param_KHP[3]
        array_KHP[18] = self.param_KHP[4]
        array_KHP[19] = self.param_KHP[4]
        array_KHP[20] = self.param_KHP[4]
        array_KHP[21] = self.param_KHP[4]
        array_KHP[22] = self.param_KHP[4]
        array_KHP[23] = self.param_KHP[4]
        array_KHP[24] = self.param_KHP[5]
        array_KHP[25] = self.param_KHP[5]
        array_KHP[26] = self.param_KHP[5]
        array_KHP[27] = self.param_KHP[5]
        array_KHP[28] = self.param_KHP[5]
        array_KHP[29] = self.param_KHP[5]
        array_KHP[30] = self.param_KHP[5]
        array_KHP[31] = self.param_KHP[5]
        array_KHP[32] = self.param_KHP[5]
        array_KHP[33] = self.param_KHP[6]
        array_KHP[34] = self.param_KHP[6]
        array_KHP[35] = self.param_KHP[6]
        array_KHP[36] = self.param_KHP[6]
        array_KHP[37] = self.param_KHP[6]
        array_KHP[38] = self.param_KHP[6]
        array_KHP[39] = self.param_KHP[6]
        array_KHP[40] = self.param_KHP[6]
        array_KHP[41] = self.param_KHP[6]
        array_KHP[42] = self.param_KHP[6]
        array_KHP[43] = self.param_KHP[6]
        array_KHP[44] = self.param_KHP[6]
        array_KHP[45] = self.param_KHP[6]
        array_KHP[46] = self.param_KHP[6]
        array_KHP[47] = self.param_KHP[6]
        array_KHP[48] = self.param_KHP[6]
        array_KHP[49] = self.param_KHP[6]

        YieldStress = f3(f2(f1(self.subModel1(LSR_input), Temp_input, Srate_input, array_deltaH)), array_KHP, GrainSize_input)
        return YieldStress

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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
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











