#*********************************************************************************************************************
# This file trains the model to fit the strength-LSR relation
# Written by Wu-Rong Jian to process the data about LSR
# Usage: python3 Pytorch_fitting_strength_LSR.py
#*********************************************************************************************************************
import os, sys, argparse, warnings, numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator
import util_fitting
torch.manual_seed(0); np.random.seed(0)
warnings.filterwarnings("ignore", category=UserWarning)

class StrengthTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        self.device = torch.device(f"cuda:{args.cuda_idx}" if (torch.cuda.is_available() and args.cuda) else "cpu")
        util_fitting.device = self.device
        print(f"Using device: {self.device}")
        os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.fig_dir, exist_ok=True)
        self.model_path = os.path.join(args.out_dir, "Model.pth")
        self.model = util_fitting.CustomModel().to(self.device).double()
        self.criterion = nn.MSELoss()
        self.optimizer = {"SGD": optim.SGD, "Adam": optim.Adam}[args.optimizer](self.model.parameters(), lr=args.lr)
        self.loss_hist = []
        self._load_data()

    def _load_txt_col1(self, path): return np.loadtxt(path, usecols=[1])

    def _load_data(self):
        d = self.args.data_dir
        ys_np = self._load_txt_col1(os.path.join(d, "YieldStress4strength.txt"))
        gs_np = self._load_txt_col1(os.path.join(d, "GrainSize4strength.txt"))
        sr_np = self._load_txt_col1(os.path.join(d, "StrainRate4strength.txt"))
        tp_np = self._load_txt_col1(os.path.join(d, "Temp4strength.txt"))
        parts = [("edge110","screw110"), ("edge112","screw112"), ("edge123","screw123")]
        cols = []
        for e, s in parts:
            cols += [self._load_txt_col1(os.path.join(d, f"{e}LSR4strength.txt")),
                     self._load_txt_col1(os.path.join(d, f"{s}LSR4strength.txt"))]
        lsr_np = np.column_stack(cols)

        self.Y_exp_np = ys_np.copy()
        self.Y_exp = torch.tensor(ys_np, dtype=torch.float64, device=self.device)
        self.GS = torch.tensor(gs_np, dtype=torch.float64, device=self.device)
        self.SR = torch.tensor(sr_np, dtype=torch.float64, device=self.device)
        self.TP = torch.tensor(tp_np, dtype=torch.float64, device=self.device)
        self.LSR = torch.tensor(lsr_np, dtype=torch.float64, device=self.device)
        print("LSR shape:", self.LSR.shape, "| Temp:", self.TP.shape, "| StrainRate:", self.SR.shape, "| GrainSize:", self.GS.shape)

    def train(self):
        pbar = tqdm(range(self.args.epochs), ncols=100, desc="Training")
        for ep in pbar:
            self.model.train(); self.optimizer.zero_grad()
            pred = self.model(self.LSR, self.TP, self.SR, self.GS)
            loss = self.criterion(pred, self.Y_exp)
            loss.backward(); self.optimizer.step()
            self.loss_hist.append(float(loss.item()))
            if (ep+1) % self.args.log_int == 0 or ep == 0:
                tqdm.write(f"Epoch [{ep+1}/{self.args.epochs}], deltaH: {self.model.param_deltaH.detach().cpu().numpy().round(4)}, KHP: {self.model.param_KHP.detach().cpu().numpy().round(2)}, Loss: {loss.item():.6f}")
                pbar.set_postfix(loss=f"{loss.item():.6f}")
        self._save_losses()

    def _save_losses(self):
        f = os.path.join(self.args.out_dir, "loss.txt")
        np.savetxt(f, np.asarray(self.loss_hist))
        print(f"---> saved data: {f}")

    def save_artifacts(self):
        # ----- per-sample a_ij(z) -----
        self.model.eval()
        with torch.no_grad():
            # Build the same context z used inside the model
            z = self.model._build_context(self.TP, self.SR, self.GS)              # (N,3)
            A = self.model.subModel1.a_layer(z).detach().cpu().numpy()            # (N,6,6)
        # Save full matrices and upper-tri vectors
        fA = os.path.join(self.args.out_dir, "aij_matrices.npy")
        np.save(fA, A); print(f"---> saved data: {fA}")
        iu = np.triu_indices(6)
        A_tri = A[:, iu[0], iu[1]]                                                # (N,21)
        fAT = os.path.join(self.args.out_dir, "aij_upper_tri.npy")
        np.save(fAT, A_tri); print(f"---> saved data: {fAT}")
        # Also save a quick-look mean a_ij over the dataset
        A_mean = A.mean(axis=0)                                                   # (6,6)
        fMean = os.path.join(self.args.out_dir, "aij_mean.csv")
        np.savetxt(fMean, A_mean, delimiter=",", fmt="%.6f"); print(f"---> saved data: {fMean}")
        # ----- RFF layer parameters (to reproduce a_ij(z)) -----
        al = self.model.subModel1.a_layer
        rff_params_path = os.path.join(self.args.out_dir, "aij_rff_params.npz")
        np.savez(rff_params_path,
                 log_lengthscale=al.log_lengthscale.detach().cpu().numpy(),
                 W_base=al.W_base.detach().cpu().numpy(),
                 b=al.b.detach().cpu().numpy(),
                 lin_weight=al.lin.weight.detach().cpu().numpy(),
                 lin_bias=al.lin.bias.detach().cpu().numpy(),)
        print(f"---> saved data: {rff_params_path}")
        # ----- physics parameters (unchanged) -----
        f3 = os.path.join(self.args.out_dir, "param_deltaH.txt")
        np.savetxt(f3, self.model.param_deltaH.detach().cpu().numpy(), fmt="%.4f"); print(f"---> saved data: {f3}")

        f4 = os.path.join(self.args.out_dir, "param_KHP.txt")
        np.savetxt(f4, self.model.param_KHP.detach().cpu().numpy(), fmt="%.6f"); print(f"---> saved data: {f4}")
        # ----- model state dict -----
        torch.save(self.model.state_dict(), self.model_path)
        print(f"---> saved data: {self.model_path}")

    def load_model_if_exists(self):
        if os.path.isfile(self.model_path):
            sd = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(sd)
            print(f"Loaded existing model: {self.model_path}")
            return True
        return False

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.LSR, self.TP, self.SR, self.GS).detach().cpu().numpy()
        return y_pred

    def plot_predictions(self, y_pred):
        Ncount = [1, 2, 8, 7, 6, 9, 17]
        labels = ['NbTaTi','HfNbTa','NbTiZr','HfNbTi','HfTaTi','HfNbTaTi','HfNbTaTiZr']
        colors = [f"C{i}" for i in range(len(Ncount))]
        markers = ['o','*','^','v','<','>','D']

        ys_exp_chunks, ys_pred_chunks, idx = [], [], 0
        for c in Ncount:
            ys_exp_chunks.append(self.Y_exp_np[idx:idx+c])
            ys_pred_chunks.append(y_pred[idx:idx+c]); idx += c

        # ---- overall metrics ----
        y_true_all = self.Y_exp_np
        y_pred_all = y_pred
        r2_all = r2_score(y_true_all, y_pred_all)
        eps = 1e-8
        mape_all = float(np.mean(np.abs((y_pred_all - y_true_all) / np.maximum(eps, np.abs(y_true_all)))) * 100.0)

        print(f"Overall R²: {r2_all:.4f} | MAPE: {mape_all:.2f}%")

        plt.figure(figsize=(7.5,7))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 22
        plt.rcParams['mathtext.fontset'] = 'stix'

        r2s = []
        for i in range(len(Ncount)):
            r2s.append(r2_score(ys_exp_chunks[i], ys_pred_chunks[i]))
            plt.scatter(ys_exp_chunks[i], ys_pred_chunks[i],
                        color=colors[i], marker=markers[i], label=labels[i], s=65)
        lim = [0, 1600]
        plt.plot(lim, lim, 'k--', linewidth=2)
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, top=True, right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        plt.xlabel('Experimental yield stress (MPa)')
        plt.ylabel('Predicted yield stress (MPa)')
        plt.xlim(lim); plt.ylim(lim)
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5), frameon=False)
        for s in ax.spines.values(): s.set_linewidth(2)

        # ---- annotate metrics on top of the plot ----
        txt = f"R$^2$ = {r2_all:.3f}\nMAPE = {mape_all:.2f}%"
        ax.text(0.1, 0.9, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

        fpng = os.path.join(self.args.fig_dir, "Comparison_yield_stress.png")
        fpdf = os.path.join(self.args.fig_dir, "Comparison_yield_stress.pdf")
        plt.savefig(fpng, bbox_inches='tight'); print(f"---> saved figure: {fpng}")
        plt.savefig(fpdf, bbox_inches='tight'); print(f"---> saved figure: {fpdf}")

        f_r2 = os.path.join(self.args.out_dir, "r2_MPEAs.txt")
        np.savetxt(f_r2, np.asarray(r2s), fmt="%.6f"); print(f"---> saved data: {f_r2}")

    def plot_loss(self):
        if len(self.loss_hist) == 0:
            loss_file = os.path.join(self.args.out_dir, "loss.txt")
            if os.path.isfile(loss_file):
                self.loss_hist = np.loadtxt(loss_file).tolist()
        if len(self.loss_hist) == 0: return
        plt.figure(figsize=(7.5,5))
        plt.rcParams['font.family'] = 'Times New Roman'; plt.rcParams['font.size'] = 22; plt.rcParams['mathtext.fontset'] = 'stix'
        plt.semilogy(self.loss_hist, color='darkblue', label='Training', linewidth=2.5)
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.tick_params(axis='both', which='both', direction='in'); plt.legend(loc="upper right", frameon=False)
        f = os.path.join(self.args.fig_dir, "loss.png")
        plt.savefig(f, bbox_inches='tight'); print(f"---> saved figure: {f}")

    def plot_aij_components(self):
        """Visualize fitted a_ij as (1) mean 6x6 heatmap and (2) 21-comp mean±std bar chart."""
        self.model.eval()
        with torch.no_grad():
            # Build context and evaluate a_ij(z) for all samples
            z = self.model._build_context(self.TP, self.SR, self.GS)      # (N,3)
            A = self.model.subModel1.a_layer(z)                           # (N,6,6)
            # Ensure exact symmetry (model should output symmetric, but make it explicit)
            A = 0.5 * (A + A.transpose(-1, -2))
            A_np = A.detach().cpu().numpy()                                # (N,6,6)

        # ---- Figure 1: mean heatmap of a_ij ----
        A_mean = A_np.mean(axis=0)                                         # (6,6)
        plt.figure(figsize=(6.5, 5.6))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['mathtext.fontset'] = 'stix'

        v = np.max(np.abs(A_mean))
        im = plt.imshow(A_mean, origin='upper', cmap='coolwarm', vmin=-v, vmax=v)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=16)

        slip_labels = [r'$\gamma^{110}$', r'$\tau^{110}$', r'$\gamma^{112}$',
                       r'$\tau^{112}$', r'$\gamma^{123}$', r'$\tau^{123}$']
        plt.xticks(range(6), slip_labels, rotation=45)
        plt.yticks(range(6), slip_labels)
        ax = plt.gca()
        for s in ax.spines.values(): s.set_linewidth(2)
        ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, top=True, right=True)

        # annotate values
        for i in range(6):
            for j in range(6):
                plt.text(j, i, f"{A_mean[i,j]:.2f}", ha='center', va='center', fontsize=15, color='black')

        plt.title(r"Mean $a_{ij}$ over dataset")
        fpng = os.path.join(self.args.fig_dir, "aij_mean_heatmap.png")
        fpdf = os.path.join(self.args.fig_dir, "aij_mean_heatmap.pdf")
        plt.tight_layout(); plt.savefig(fpng, dpi=300); print(f"---> saved figure: {fpng}")
        plt.savefig(fpdf); print(f"---> saved figure: {fpdf}")

        # ---- Figure 2: upper-triangular components (mean ± std) ----
        iu = np.triu_indices(6) # (21,)
        A_tri = A_np[:, iu[0], iu[1]] # (N,21)
        mu = A_tri.mean(axis=0)
        sd = A_tri.std(axis=0)

        # human-friendly labels like "e110–e110", "e110–s110", ...
        pair_labels = [rf"{slip_labels[i]}$\leftrightarrow${slip_labels[j]}" for i, j in zip(iu[0], iu[1])]

        plt.figure(figsize=(12, 5))
        # plt.rcParams['font.family'] = 'Times New Roman'
        # plt.rcParams['font.size'] = 16
        # plt.rcParams['mathtext.fontset'] = 'stix'

        x = np.arange(len(mu))
        plt.bar(x, mu, yerr=sd, capsize=3, color='skyblue', edgecolor='black', error_kw={'elinewidth':1.5})
        ax = plt.gca(); ax.set_xticks(x)
        # show every label but keep it readable by tilting; feel free to thin ticks if crowded
        ax.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax.set_ylabel(r"$a_{ij}$ (mean $\pm$ std)")
        ax.set_title(r"Upper-triangular components of $a_{ij}$")
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        for s in ax.spines.values(): s.set_linewidth(2)

        fpng = os.path.join(self.args.fig_dir, "aij_upper_tri_bar.png")
        fpdf = os.path.join(self.args.fig_dir, "aij_upper_tri_bar.pdf")
        plt.tight_layout(); plt.savefig(fpng, dpi=300); print(f"---> saved figure: {fpng}")
        plt.savefig(fpdf); print(f"---> saved figure: {fpdf}")


def parse_args():
    p = argparse.ArgumentParser(description="Fit CustomModel to strength data and plot results.")
    p.add_argument("--data_dir", default="data"); p.add_argument("--out_dir", default="outputs"); p.add_argument("--fig_dir", default="figures")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--optimizer", choices=["SGD","Adam"], default="Adam")
    p.add_argument("--seed", type=int, default=21); p.add_argument("--log_int", type=int, default=100)
    p.add_argument("--cuda", action="store_true"); p.add_argument("--cuda_idx", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = StrengthTrainer(args)
    if not trainer.load_model_if_exists():
        trainer.train()
        trainer.save_artifacts()
    y_pred = trainer.predict()
    trainer.plot_predictions(y_pred)
    trainer.plot_loss()
    trainer.plot_aij_components() 