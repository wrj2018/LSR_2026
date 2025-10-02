import os, warnings, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator
import util_model as um

torch.manual_seed(0); np.random.seed(0)
warnings.filterwarnings("ignore", category=UserWarning)

def _latex_pm(mean, std, fmt="{:.4f}"):
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"

class StrengthTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # propagate chosen device to model utilities
        um.device = self.device
        print(f"Using device: {self.device}")
        os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.fig_dir, exist_ok=True)
        self.model_path = os.path.join(args.out_dir, "Model.pth")
        self.best_model_path = os.path.join(args.out_dir, "Model_best.pth")
        self.model = um.CustomModel().to(self.device).double()
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
        best_loss = float('inf')
        for ep in pbar:
            self.model.train(); self.optimizer.zero_grad()
            pred = self.model(self.LSR, self.TP, self.SR, self.GS)
            loss = self.criterion(pred, self.Y_exp)
            loss.backward(); self.optimizer.step()
            self.loss_hist.append(float(loss.item()))
            # Save best model when loss improves
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), self.best_model_path)
                # if loss.item() <= 100:
                #     tqdm.write(f"[Best@{ep+1}] Loss improved to {best_loss:.6f}; saved: {self.best_model_path}")
            if (ep+1) % self.args.log_int == 0 or ep == 0:
                tqdm.write(f"Epoch [{ep+1}/{self.args.epochs}], Loss: {loss.item():.6f}")
                # deltaH: {self.model.param_deltaH.detach().cpu().numpy().round(4)}, KHP: {self.model.param_KHP.detach().cpu().numpy().round(2)}, 
                pbar.set_postfix(loss=f"{loss.item():.6f}")
        # Load best weights back into the model for downstream artifacts/plots
        if os.path.isfile(self.best_model_path):
            sd_best = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(sd_best)
            tqdm.write(f"Loaded best model for evaluation: {self.best_model_path}")
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
        # ----- model state dict (save current model as standard path) -----
        # At this point, self.model holds the best weights (loaded after training).
        torch.save(self.model.state_dict(), self.model_path)
        print(f"---> saved data: {self.model_path}")
        # Also ensure best checkpoint exists (already saved during training). Duplicate if not.
        if not os.path.isfile(self.best_model_path):
            torch.save(self.model.state_dict(), self.best_model_path)
        print(f"---> best model path: {self.best_model_path}")

    def load_model_if_exists(self):
        # Prefer loading best checkpoint if available
        ckpt_path = None
        if os.path.isfile(self.best_model_path):
            ckpt_path = self.best_model_path
        elif os.path.isfile(self.model_path):
            ckpt_path = self.model_path
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(sd)
            print(f"Loaded existing model: {ckpt_path}")
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
        """Visualize fitted a_ij as (1) mean 6x6 heatmap and (2) 21-comp mean±std bar chart,
        and also WRITE LaTeX/CSV tables for the 21 upper-tri components (mean ± std)."""
        self.model.eval()
        with torch.no_grad():
            z = self.model._build_context(self.TP, self.SR, self.GS)  # (N,3)
            A = self.model.subModel1.a_layer(z)                       # (N,6,6)
            A = 0.5 * (A + A.transpose(-1, -2))
            A_np = A.detach().cpu().numpy()                           # (N,6,6)
        # ---- Figure 1: mean heatmap of a_ij ----
        A_mean = A_np.mean(axis=0)                                    # (6,6)
        plt.figure(figsize=(6.5, 5.6))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['mathtext.fontset'] = 'stix'
        v = np.max(np.abs(A_mean))
        im = plt.imshow(A_mean, origin='upper', cmap='coolwarm', vmin=-v, vmax=v)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=16)

        slip_labels = ['$\gamma^{110}$', '$\tau^{110}$', '$\gamma^{112}$',
                       '$\tau^{112}$', '$\gamma^{123}$', '$\tau^{123}$']
        plt.xticks(range(6), slip_labels, rotation=45)
        plt.yticks(range(6), slip_labels)
        ax = plt.gca()
        for s in ax.spines.values(): s.set_linewidth(2)
        ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, top=True, right=True)

        for i in range(6):
            for j in range(6):
                plt.text(j, i, f"{A_mean[i,j]:.2f}", ha='center', va='center', fontsize=15, color='black')

        plt.title(r"Mean $a_{ij}$ over dataset")
        fpng = os.path.join(self.args.fig_dir, "aij_mean_heatmap.png")
        fpdf = os.path.join(self.args.fig_dir, "aij_mean_heatmap.pdf")
        plt.tight_layout(); plt.savefig(fpng, dpi=300); print(f"---> saved figure: {fpng}")
        plt.savefig(fpdf); print(f"---> saved figure: {fpdf}")

        # ---- Figure 2 + tables: upper-tri (mean ± std)
        iu = np.triu_indices(6)                # (21,)
        A_tri = A_np[:, iu[0], iu[1]]          # (N,21)
        mu = A_tri.mean(axis=0)                # (21,)
        sd = A_tri.std(axis=0)                 # (21,)

        pair_labels = [rf"{slip_labels[i]}$\leftrightarrow${slip_labels[j]}" for i, j in zip(iu[0], iu[1])]

        plt.figure(figsize=(12, 5))
        x = np.arange(len(mu))
        plt.bar(x, mu, yerr=sd, capsize=3, color='skyblue', edgecolor='black', error_kw={'elinewidth':1.5})
        ax = plt.gca(); ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax.set_ylabel(r"$a_{ij}$ (mean $\pm$ std)")
        ax.set_title(r"Upper-triangular components of $a_{ij}$")
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        for s in ax.spines.values(): s.set_linewidth(2)

        fpng = os.path.join(self.args.fig_dir, "aij_upper_tri_bar.png")
        fpdf = os.path.join(self.args.fig_dir, "aij_upper_tri_bar.pdf")
        plt.tight_layout(); plt.savefig(fpng, dpi=300); print(f"---> saved figure: {fpng}")
        plt.savefig(fpdf); print(f"---> saved figure: {fpdf}")

        # ---- Write CSV + LaTeX table for mean ± std of the 21 comps
        stats_csv = os.path.join(self.args.out_dir, "aij_upper_tri_stats.csv")
        with open(stats_csv, "w") as f:
            f.write("pair,mean,std\n")
            for lbl, m, s in zip(pair_labels, mu, sd):
                f.write(f"{lbl},{m:.6f},{s:.6f}\n")
        print(f"---> wrote A_ij stats CSV: {stats_csv}")
        stats_tex = os.path.join(self.args.out_dir, "aij_upper_tri_table.tex")
        with open(stats_tex, "w") as f:
            f.write("% Mean ± std for upper-triangular a_ij components (21 entries)\n")
            f.write("\\begin{tabular}{lcc}\\hline\nPair & Mean & Std \\\\ \\hline\n")
            for lbl, m, s in zip(pair_labels, mu, sd):
                f.write(f"{lbl} & {m:.4f} & {s:.4f} \\\\ \n")
            f.write("\\hline\\end{tabular}\n")
        print(f"---> wrote A_ij LaTeX table: {stats_tex}")

    def plot_aij_per_material(self):
        """For each material (composition), print and save heatmaps of a_ij for every sample."""
        self.model.eval()
        with torch.no_grad():
            z = self.model._build_context(self.TP, self.SR, self.GS)  # (N,3)
            A = self.model.subModel1.a_layer(z)                       # (N,6,6)
            A = 0.5 * (A + A.transpose(-1, -2))
            A_np = A.detach().cpu().numpy()                           # (N,6,6)

        labels = ['NbTaTi','HfNbTa','NbTiZr','HfNbTi','HfTaTi','HfNbTaTi','HfNbTaTiZr']
        counts = [1, 2, 8, 7, 6, 9, 17]
        # slip_labels = ['$\gamma^{110}$', '$\tau^{110}$', '$\gamma^{112}$',
        #                '$\tau^{112}$', '$\gamma^{123}$', '$\tau^{123}$']
        slip_labels = ['1', '2', '3', '4', '5', '6']

        start = 0
        for name, cnt in zip(labels, counts):
            end = start + cnt
            mats = A_np[start:end]
            for k, M in enumerate(mats, start=1):
                print(f"\n=== a_ij for {name}, sample {k}/{cnt} ===")
                for i in range(6):
                    row = " ".join(f"{M[i,j]:.2f}" for j in range(6))
                    print(row)
                plt.figure(figsize=(6.5, 5.6))
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.size'] = 20
                plt.rcParams['mathtext.fontset'] = 'stix'
                v = np.max(np.abs(M))
                im = plt.imshow(M, origin='upper', cmap='coolwarm', vmin=0, vmax=v)
                cb = plt.colorbar(im, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=16)
                plt.xticks(range(6), slip_labels, rotation=45)
                plt.yticks(range(6), slip_labels)
                ax = plt.gca()
                for s in ax.spines.values(): s.set_linewidth(2)
                ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, top=True, right=True)
                for i in range(6):
                    for j in range(6):
                        plt.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center', fontsize=15, color='black')
                plt.title(rf"$a_{{ij}}$ — {name} (sample {k}/{cnt})")
                fpng = os.path.join(self.args.fig_dir, f"aij_{name}_s{k}.png")
                fpdf = os.path.join(self.args.fig_dir, f"aij_{name}_s{k}.pdf")
                plt.tight_layout(); plt.savefig(fpng, dpi=300); print(f"---> saved figure: {fpng}")
                plt.savefig(fpdf); print(f"---> saved figure: {fpdf}")
                plt.close()
            start = end

    def print_aij_mean_std_table(self):
        """Print and export LaTeX for mean (std) of the fitted 6x6 a_ij over the dataset."""
        self.model.eval()
        with torch.no_grad():
            # Build context and evaluate a_ij(z) for all samples
            z = self.model._build_context(self.TP, self.SR, self.GS)   # (N,3)
            A = self.model.subModel1.a_layer(z)                        # (N,6,6)
            A = 0.5 * (A + A.transpose(-1, -2))                        # enforce symmetry
            A_np = A.detach().cpu().numpy()                            # (N,6,6)

        mu = A_np.mean(axis=0)  # (6,6)
        sd = A_np.std(axis=0)   # (6,6)

        # ---------- Console pretty print ----------
        def cell(m, s): return f"{m:.2f} ({s:.2f})"
        hdr = ["    a_ij  "] + [f" j={j+1}" for j in range(6)]
        print("\n=== Fitted a_ij mean (std) over dataset ===")
        print(" | ".join(hdr))
        print("-" * (12 + 9*6))
        for i in range(6):
            row_vals = [cell(mu[i, j], sd[i, j]) for j in range(6)]
            print(f" i={i+1}  | " + " | ".join(f"{v:>9s}" for v in row_vals))

        # ---------- Also save CSVs for mean and std (optional but handy) ----------
        os.makedirs(self.args.out_dir, exist_ok=True)
        mean_csv = os.path.join(self.args.out_dir, "aij_mean.csv")
        std_csv  = os.path.join(self.args.out_dir, "aij_std.csv")
        np.savetxt(mean_csv, mu, delimiter=",", fmt="%.6f")
        np.savetxt(std_csv,  sd, delimiter=",", fmt="%.6f")
        print(f"---> saved data: {mean_csv}")
        print(f"---> saved data: {std_csv}")

        # ---------- LaTeX table ----------
        # Use simple i/j indices in headers; swap with your slip labels if preferred.
        latex_lines = []
        latex_lines += [
            r"\begin{table*}[htp]",
            r"\caption{Mean (std) of fitted interaction matrix $a_{ij}(z)$ over all samples.}",
            r"\label{tbl:aij_mean_std}",
            r"\centering",
            r"\setlength{\tabcolsep}{6pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{ccccccc}",
            r"\hline",
            r"$i\backslash j$ & 1 & 2 & 3 & 4 & 5 & 6 \\ \hline",
        ]
        for i in range(6):
            row_strs = [f"{mu[i,j]:.2f} ({sd[i,j]:.2f})" for j in range(6)]
            latex_lines.append(f"{i+1} & " + " & ".join(row_strs) + r" \\")
        latex_lines += [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            ""
        ]
        tex_path = os.path.join(self.args.out_dir, "aij_mean_std_table.tex")
        with open(tex_path, "w") as f:
            f.write("\n".join(latex_lines))
        print(f"---> wrote LaTeX table: {tex_path}")

    def print_fitted_params(self):
        """Print fitted K_HP and ΔH0 for quick visual check; also write LaTeX snippets."""
        self.model.eval()
        with torch.no_grad():
            K = self.model.param_KHP.detach().cpu().numpy()          # (7,)
            DH = self.model.param_deltaH.detach().cpu().numpy()      # (7,6)

        comp_labels = ['NbTaTi','HfNbTa','NbTiZr','HfNbTi','HfTaTi','HfNbTaTi','HfNbTaTiZr']
        slip_order  = [r'$\gamma^{110}$', r'$\tau^{110}$', r'$\gamma^{112}$',
                       r'$\tau^{112}$', r'$\gamma^{123}$', r'$\tau^{123}$']

        # ---- Console print: K_HP ----
        print("\n=== Fitted K_HP (MPa·μm^{-1/2}) ===")
        for name, val in zip(comp_labels, K):
            print(f"{name:>12s}: {val:.4f}")

        # ---- Console print: ΔH0 ----
        print("\n=== Fitted ΔH0 (eV) per composition × slip system ===")
        print("cols order:", " | ".join(slip_order))
        for name, row in zip(comp_labels, DH):
            print(f"{name:>12s}: " + "  ".join(f"{v:.4f}" for v in row))

        # ---- Write quick LaTeX snippets you can paste ----
        os.makedirs(self.args.out_dir, exist_ok=True)
        # Table A1 one-line row for K_HP
        f1 = os.path.join(self.args.out_dir, "KHP_table_row.tex")
        with open(f1, "w") as f:
            f.write("% Paste into Table A1 body\n")
            f.write("$K_\\mathrm{HP}$ & " + " & ".join(f"{v:.2f}" for v in K) + " \\\\ \\hline\n")
        print(f"---> wrote LaTeX row for K_HP: {f1}")

        # Table A2 body for ΔH0
        f2 = os.path.join(self.args.out_dir, "DeltaH_table_body.tex")
        with open(f2, "w") as f:
            f.write("% Paste rows into Table A2 body\n")
            for name, row in zip(comp_labels, DH):
                f.write(f"{name} & " + " & ".join(f"{v:.4f}" for v in row) + " \\\\ \\hline\n")
        print(f"---> wrote LaTeX rows for ΔH0: {f2}")
