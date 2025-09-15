# Imports
import torch
import torch.nn as nn
import numpy as np
import os
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Functions #
def f1(LSR_input, Temp_input, Srate_input, deltaH_input):
    # print("f1_LSR_input shape:", LSR_input.shape)
    # print("f1_Temp_input shape:", Temp_input.shape)
    # print("f1_Srate_input shape:", Srate_input.shape)
    # print("f1_deltaH_input shape:", deltaH_input.shape)

    Temp_input_expanded = Temp_input.unsqueeze(1).expand(-1, 6)  # Shape becomes [50, 6]
    Srate_input_expanded = Srate_input.unsqueeze(1).expand(-1, 6)  # Shape becomes [50, 6]

    # Boltzmann_constant; Unit: eV/K
    Kb = torch.tensor([8.62e-5], dtype=torch.float64, device=device)
    # Reference strain rate; Unit: s^{-1}
    epsilon0_dot = torch.tensor([10000], dtype=torch.float64, device=device)
    output_tensor = LSR_input*(1-torch.pow((Kb*Temp_input_expanded*torch.log(epsilon0_dot/Srate_input_expanded)/deltaH_input), 2/3))
    return output_tensor

def f2(input_tensor):
    # print("f2_input shape:", input_tensor.shape)
    output_tensor = torch.sum(input_tensor, dim=1)
    # print("f2_output shape:", output_tensor.shape)
    return output_tensor

def f3(input_tensor, KHP_input, GrainSize_input):
    # print("f3_input shape:", input_tensor.shape)
    # print("f3_KHP_input shape:", KHP_input.shape)
    # print("f3_GrainSize_input shape:", GrainSize_input.shape)
    
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