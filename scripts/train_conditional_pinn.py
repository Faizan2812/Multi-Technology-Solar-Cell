"""
train_conditional_pinn.py — train and validate the PRODUCTION conditional PINN.

Trains ONE network across a (d_abs, Nt) device family using production
drift-diffusion anchor solutions + autograd PDE residuals, then evaluates on
UNSEEN interpolation devices. Saves model weights and the validation metrics
to artifacts/.

Usage:
    python scripts/train_conditional_pinn.py
    python scripts/train_conditional_pinn.py --epochs-data 2000 --epochs-pde 3000
"""
import os, sys, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-data", type=int, default=1500)
    ap.add_argument("--epochs-pde", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from ai.conditional_pinn_torch import (DeviceFamily, train_conditional_pinn,
                                           evaluate_interpolation)

    h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
    fam = DeviceFamily(h, a, e)

    print("Training conditional PINN on 4 DD anchor devices "
          f"(corners of d_abs in [{fam.d_lo:.0f},{fam.d_hi:.0f}] nm x "
          f"Nt in [1e{fam.n_lo:.0f},1e{fam.n_hi:.0f}] cm^-3)...")
    model, hist = train_conditional_pinn(
        fam, epochs_data=args.epochs_data, epochs_pde=args.epochs_pde,
        lr=args.lr, seed=args.seed, verbose=True)

    unseen = [(500.0, 1e14), (400.0, 3e15), (600.0, 1e15)]
    metrics = {}
    print("\nInterpolation validation on UNSEEN devices:")
    for d, Nt in unseen:
        ev = evaluate_interpolation(model, fam, d, Nt)
        key = f"d{d:.0f}_Nt{Nt:.0e}"
        metrics[key] = {"relL2_psi": ev["relL2_psi"],
                        "relL2_logn": ev["relL2_logn"]}
        print(f"  (d={d:.0f} nm, Nt={Nt:.0e}): "
              f"relL2 psi = {ev['relL2_psi']:.4f}, "
              f"relL2 log n = {ev['relL2_logn']:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/conditional_pinn.pt")
    with open("artifacts/conditional_pinn_metrics.json", "w") as f:
        json.dump({
            "family": {"htl": "Spiro-OMeTAD", "abs": "MAPbI3", "etl": "SnO2",
                       "d_abs_range_nm": [fam.d_lo, fam.d_hi],
                       "logNt_range": [fam.n_lo, fam.n_hi]},
            "training": {"epochs_data": args.epochs_data,
                         "epochs_pde": args.epochs_pde, "lr": args.lr,
                         "seed": args.seed,
                         "final_loss_data": hist["loss_data"][-1],
                         "final_loss_pde": (hist["loss_pde"][-1]
                                            if hist["loss_pde"] else None)},
            "interpolation_validation": metrics,
        }, f, indent=2)
    with open("artifacts/conditional_pinn_history.pkl", "wb") as f:
        pickle.dump(hist, f)
    print("\nSaved: artifacts/conditional_pinn.pt, "
          "artifacts/conditional_pinn_metrics.json")


if __name__ == "__main__":
    main()
