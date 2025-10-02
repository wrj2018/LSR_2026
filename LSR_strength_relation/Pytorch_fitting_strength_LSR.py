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
    trainer.print_aij_mean_std_table()
    trainer.print_fitted_params()