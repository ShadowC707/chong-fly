import os
import glob
import numpy as np
import scipy.sparse as sp

def compute_spectral_radius(w):
    if sp.issparse(w):
        w_dense = w.toarray()
    else:
        w_dense = w
    eigenvalues = np.linalg.eigvals(w_dense)
    return np.max(np.abs(eigenvalues))

def main():
    model_dir = "data/reduced_models"
    files = glob.glob(os.path.join(model_dir, "w_*.*"))
    for file in sorted(files):
        if file.endswith('.npy'):
            w = np.load(file)
        elif file.endswith('.npz'):
            w = sp.load_npz(file)
        else:
            continue
        
        rho = compute_spectral_radius(w)
        print(f"{os.path.basename(file)}: rho(W) = {rho:.4f}")

if __name__ == "__main__":
    main()
