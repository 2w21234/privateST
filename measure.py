import os
import sys
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# Ignore runtime warnings for clean output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
# Path to the Ground Truth (GT) file and results directory
TARGET_PATH = 'model/epoch_15.npz'
PRED_DIR = "results/"

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def calculate_metrics(yt, yp, top_k):
    """
    Calculates Pearson/Spearman correlation and RMSE for each gene.
    """
    pccs, sccs, rmses = [], [], []
    for g in range(top_k):
        t = yt[:, g]
        p = yp[:, g]



        # Avoid calculation errors if standard deviation is zero
        if np.std(t) > 1e-9 and np.std(p) > 1e-9:
            pc, _ = pearsonr(t, p)
            sc, _ = spearmanr(t, p)
        else:
            pc, sc = 0.0, 0.0

        rmse = np.sqrt(mean_squared_error(t, p))
        pccs.append(pc)
        sccs.append(sc)
        rmses.append(rmse)

    return np.mean(pccs), np.mean(sccs), np.mean(rmses)

def load_predictions(prefix, sections, coords, yt_shape, top_indices, top_k):
    """
    Searches for and loads individual .npy prediction files based on prefix.
    """
    yp_list = []
    found_count = 0

    for i in range(len(sections)):
        sec = sections[i]
        if hasattr(sec, 'item'): sec = sec.item()
        if isinstance(sec, bytes): sec = sec.decode('utf-8')
        x, y = coords[i]

        # File naming convention: Approx_D1_8_18.npy or HE_D1_8_18.npy
        fname = f"{prefix}_{sec}_{int(x)}_{int(y)}.npy"
        fpath = os.path.join(PRED_DIR, fname)

        if os.path.exists(fpath):
            try:
                arr = np.load(fpath).flatten()
                # Ensure predicted gene count matches GT before slicing
                if arr.shape[0] >= yt_shape[1]:
                    arr_sliced = arr[:yt_shape[1]]
                    yp_list.append(arr_sliced[top_indices])
                    found_count += 1
                else:
                    yp_list.append(np.zeros(top_k))
            except Exception:
                yp_list.append(np.zeros(top_k))
        else:
            yp_list.append(np.zeros(top_k))

    return np.array(yp_list), found_count

# -----------------------------------------------------------------------------
# 3. Main Evaluation Logic
# -----------------------------------------------------------------------------
def run_evaluation():
    # --- A. Load Ground Truth (GT) Data ---
    if not os.path.exists(TARGET_PATH):
        print(f"[ERROR] GT file not found: {TARGET_PATH}")
        sys.exit(1)

    z = np.load(TARGET_PATH, allow_pickle=True)
    yt_base = z['counts'] if 'counts' in z else z['true_count']
    sections = z['section'] if 'section' in z else z['sections']
    coords = z['index'] if 'index' in z else z['coord']

    print(f"[INFO] GT Data Loaded: {yt_base.shape} (Spots x Genes)")

    # --- B. Gene Ranking (Top 100 by Mean Expression) ---
    means = np.mean(yt_base, axis=0)
    sort_idx = np.argsort(means)[::-1]
    top_k = 100
    top_indices = sort_idx[:top_k]
    yt_top100 = yt_base[:, top_indices]

    # --- C. Run Metrics for Each Mode (Approx and HE) ---
    modes = ["Approx", "HE"]

    for mode in modes:
        # Load and aggregate data
        yp_top100, found = load_predictions(mode, sections, coords, yt_base.shape, top_indices, top_k)

        if found == 0:
            continue # Skip if no files found for this mode

        # Save the resulting top 100 prediction matrix
        save_name = f"results/top100_preds_{mode}.npy"
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, yp_top100)

        # Calculate final metrics
        apcc, ascc, armse = calculate_metrics(yt_top100, yp_top100, top_k)

        print("\n" + "="*50)
        print(f" Evaluation Mode: {mode} (Files: {found}/{len(sections)})")
        print("-" * 50)
        print(f" Avg Pearson (aPCC)  : {apcc:.4f}")
        print(f" Avg Spearman (aSCC) : {ascc:.4f}")
        print(f" Avg RMSE (aRMSE)    : {armse:.4f}")
        print(f" Matrix saved to     : {save_name}")

    print("="*50)

if __name__ == "__main__":
    run_evaluation()                      
