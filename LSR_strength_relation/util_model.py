import torch, torch.nn as nn, math
# ----------------------- device / dtype -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float64

# ----------------------- physics blocks -----------------------
def f1(LSR_input, Temp_input, Srate_input, deltaH_input):
    """
    LSR_input: (B,6)   after a_ij(z) mixing
    Temp_input: (B,)
    Srate_input: (B,)
    deltaH_input: (B,6)
    Returns: (B,6)
    """
    Temp_input_expanded = Temp_input.unsqueeze(1).expand(-1, 6)  # (B,6)
    Srate_input_expanded = Srate_input.unsqueeze(1).expand(-1, 6)  # (B,6)

    # Boltzmann constant in eV/K
    Kb = torch.tensor([8.62e-5], dtype=_DTYPE, device=device)
    # Reference strain rate (s^-1)
    epsilon0_dot = torch.tensor([1.0e4], dtype=_DTYPE, device=device)

    # Guard for log and power numeric stability
    ratio = (epsilon0_dot / Srate_input_expanded.clamp_min(1e-30))           # positive
    act = (Kb * Temp_input_expanded * torch.log(ratio) / deltaH_input)       # can be negative; guard for pow
    act_pos = act.clamp_min(1e-12)                                           # ensure >=0 for fractional power
    factor = 1.0 - torch.pow(act_pos, 2.0/3.0)

    output_tensor = LSR_input * factor
    return output_tensor

def f2(input_tensor):
    """
    Sum over slip systems
    input_tensor: (B,6) -> (B,)
    """
    return torch.sum(input_tensor, dim=1)

def f3(input_tensor, KHP_input, GrainSize_input):
    """
    input_tensor: (B,)  accumulated thermal/resistance term
    KHP_input:    (B,)
    GrainSize_input: (B,)

    Returns: YieldStress (B,)
    """
    # Taylor factor for BCC metals with random texture
    param_M = torch.tensor([2.733], dtype=_DTYPE, device=device)
    YieldStress = input_tensor * param_M + KHP_input * torch.pow(GrainSize_input, -0.5)
    return YieldStress

# ----------------------- helpers for symmetric 6x6 -----------------------
def _tri_u_indices(n=6, device=None):
    return torch.triu_indices(n, n, device=device)

def unpack_sym_6x6(v):
    """ v: (..., 21) -> (..., 6, 6) symmetric """
    n = 6
    iu = _tri_u_indices(n=n, device=v.device)
    shape = v.shape[:-1]
    M = torch.zeros(*shape, n, n, dtype=v.dtype, device=v.device)
    M[..., iu[0], iu[1]] = v
    M = M + M.transpose(-1, -2) - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
    return M

# ----------------------- Gaussian-kernel (RFF) a_ij(z) layer -----------------------
class SymmetricRFFKernelLayer(nn.Module):
    """
    Random Fourier Features approximation of an RBF kernel.
    Given context z ∈ R^p, outputs a symmetric 6×6 matrix a(z).

    a(z) is produced by a linear map from features φ(z) to the 21
    upper-tri entries, then symmetrized.
    """
    def __init__(self, in_features, n_features=512, ard=True):
        super().__init__()
        self.p = in_features
        self.D = n_features
        self.ard = ard
        # Trainable lengthscales (log ℓ); ARD or shared
        if ard: self.log_lengthscale = nn.Parameter(torch.zeros(self.p, dtype=_DTYPE))
        else: self.log_lengthscale = nn.Parameter(torch.zeros(1, dtype=_DTYPE))
        # Base Gaussian matrix W_base ~ N(0, 1); scaled by 1/ℓ to realize RBF frequencies
        self.register_buffer("W_base", torch.randn(self.D, self.p, dtype=_DTYPE))
        # Random phases b ~ Uniform(0, 2π)
        self.register_buffer("b", 2 * math.pi * torch.rand(self.D, dtype=_DTYPE))
        # Linear map to 21 outputs
        self.lin = nn.Linear(self.D, 21, bias=True, dtype=_DTYPE)

    def _features(self, z):
        """
        z: (B,p) -> φ(z): (B,D)
        φ(z) = sqrt(2/D) * cos(W z + b)
        with W_d,: ~ N(0, diag(1/ℓ^2))
        """
        if self.ard:
            inv_ell = torch.exp(-self.log_lengthscale)          # (p,)
            W = self.W_base * inv_ell.unsqueeze(0)              # (D,p)
        else:
            inv_ell = torch.exp(-self.log_lengthscale)          # scalar
            W = self.W_base * inv_ell
        proj = z @ W.T + self.b                                 # (B,D)
        phi = math.sqrt(2.0 / self.D) * torch.cos(proj)
        return phi

    def forward(self, z):
        """
        z: (B,p) context -> A: (B,6,6) symmetric and strictly positive
        """
        phi = self._features(z)                  # (B,D)
        y = self.lin(phi)                        # (B,21)
        y_pos = torch.nn.functional.softplus(y)  # strictly positive
        A = unpack_sym_6x6(y_pos)                # (B,6,6)
        return A

# ----------------------- SLP: apply a_ij(z) to LSR -----------------------
class SLP(nn.Module):
    """
    Replaces the constant symmetric 6x6 with a context-dependent symmetric matrix a(z)
    computed by an RFF Gaussian-kernel layer. By default, z = [T, ln(ε̇), d].
    """
    def __init__(self, p_context=3, n_features=512, ard=True):
        super(SLP, self).__init__()
        self.a_layer = SymmetricRFFKernelLayer(in_features=p_context,
                                               n_features=n_features,
                                               ard=ard).to(device)

    def forward(self, LSR_input, context_z):
        """
        LSR_input:  (B,6)
        context_z:  (B,p_context)
        returns:    (B,6)  -> LSR' = LSR @ a(z)
        """
        A = self.a_layer(context_z)                          # (B,6,6)
        out = torch.einsum('bi,bij->bj', LSR_input, A)       # batch-right-multiply
        return out

# ----------------------- Training model -----------------------
class CustomModel(nn.Module):
    def __init__(self, p_context=3, n_features=512, ard=True):
        super(CustomModel, self).__init__()
        # a_ij(z) mixer for LSR
        self.subModel1 = SLP(p_context=p_context, n_features=n_features, ard=ard).to(device)

        # deltaH for 7 alloys × 6 systems (raw params -> constrained)
        self._raw_param_deltaH = nn.Parameter(torch.randn(7, 6, dtype=_DTYPE))
        # KHP for 7 alloys
        self._raw_param_KHP = nn.Parameter(torch.randn(7, dtype=_DTYPE))

    @property
    def param_deltaH(self):
        # Unit: eV (typical BCC range 1–3 eV). Keep (1.0, 3.0).
        # Previous: 0.3 + 2.7 * torch.sigmoid(self._raw_param_deltaH)
        return 1.5 + 1.5 * torch.sigmoid(self._raw_param_deltaH) # range (1.5, 3.0)

    @property
    def param_KHP(self):
        # Unit: MPa·µm^{-1/2}. Positive via exp.
        # return torch.exp(self._raw_param_KHP)
        # return 200 + 600*torch.sigmoid(self._raw_param_KHP)
        return 0 + 300*torch.sigmoid(self._raw_param_KHP)

    def _build_context(self, Temp_input, Srate_input, GrainSize_input):
        """
        Context z = [T, ln(ε̇), d], all as double on device.
        Shapes: each (B,) -> z: (B,3)
        """
        zT = Temp_input.to(device=device, dtype=_DTYPE)
        zSR = (Srate_input.clamp_min(1e-30)).to(device=device, dtype=_DTYPE) # torch.log
        zD = GrainSize_input.to(device=device, dtype=_DTYPE)
        z = torch.stack([zT, zSR, zD], dim=1)  # (B,3)
        return z

    def forward(self, LSR_input, Temp_input, Srate_input, GrainSize_input):
        """
        LSR_input:     (B,6)    (per-system LSR coefficients / τ_lsr,i baseline)
        Temp_input:    (B,)
        Srate_input:   (B,)
        GrainSize_input:(B,)
        """
        # ---- context for a_ij(z) ----
        context_z = self._build_context(Temp_input, Srate_input, GrainSize_input)  # (B,3)

        # ---- apply a_ij(z) on LSR ----
        LSR_mixed = self.subModel1(LSR_input, context_z)  # (B,6)

        # ---- alloy mapping (kept identical to your original code) ----
        dev = self.param_deltaH.device
        idx = torch.repeat_interleave(torch.arange(7, device=dev), torch.tensor([1, 2, 8, 7, 6, 9, 17], device=dev))
        # Expecting batch size B = 50 with the above composition layout
        self.array_deltaH = self.param_deltaH[idx]   # (B,6)
        self.array_KHP = self.param_KHP[idx]         # (B,)

        # ---- physics head ----
        out_per_sys = f1(LSR_mixed, Temp_input.to(device=device, dtype=_DTYPE),
                         Srate_input.to(device=device, dtype=_DTYPE), self.array_deltaH)
        summed = f2(out_per_sys)  # (B,)
        sigma_y = f3(summed, self.array_KHP, GrainSize_input.to(device=device, dtype=_DTYPE))
        return sigma_y
