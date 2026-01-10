#*********************************************************************************************************************
# This file trains the model to fit the strength-LSR relation
# Written by Wu-Rong Jian to process the data about LSR
# Usage: python3 Pytorch_fitting_strength_LSR.py
#*********************************************************************************************************************
import os, sys, argparse, warnings, numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator
from util_training import *
torch.manual_seed(0); np.random.seed(0)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    p = argparse.ArgumentParser(description="Fit CustomModel to strength data and plot results.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100000)
    p.add_argument("--optimizer", choices=["SGD","Adam"], default="Adam")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--log_int", type=int, default=1000)
    p.add_argument("--model_option", choices=["a_ij","a_i","a_single","a_single_const"], default="a_ij", 
                   help="Model option: 'a_ij' for 6x6 matrix mixing, 'a_i' for 6-element vector mixing, 'a_single' for context-dependent single scalar, 'a_single_const' for constant single scalar (no context)")
    p.add_argument("--load_cache", action="store_true",
                   help="Load previous model if it exists; otherwise train from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = StrengthTrainer(args)
    # Only attempt to load cached model if --load_cache flag is provided
    model_loaded = False
    if args.load_cache:
        model_loaded = trainer.load_model_if_exists()
    if not model_loaded:
        trainer.train()
        trainer.save_artifacts()
    y_pred = trainer.predict()
    trainer.plot_predictions(y_pred)
    trainer.plot_loss()
    # Plot a_ij per material and per sample instead of only averages
    try:
        # If running with util_training module naming
        trainer.plot_aij_per_material()
    except AttributeError:
        # Fallback for older versions
        trainer.plot_aij_components()
    trainer.print_aij_mean_std_table()
    trainer.print_fitted_params()
