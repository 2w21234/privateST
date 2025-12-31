import os
import numpy as np
import pandas as pd
import csv
import glob
import re
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# 1. Configuration and Path Settings
# -----------------------------------------------------------------------------
# [Ground Truth] Use 'counts' (or 'true') data from this file as the ground truth
# UPDATED: Now using the lightweight file with predictions removed
TRUE_NPZ_PATH = 'scaled_y.npz'

# [Matching CSV] Predictions must be sorted in the order of this file
NAMES_CSV_PATH = 'sample_names.csv'

# [Prediction Directory]
PRED_DIR = "results/"
# (Add additional search paths like "privateST/results/" if necessary)

# Path to save the result CSV
CSV_OUT = os.path.join(PRED_DIR, "final_evaluation_result.csv")

# Pre-load prediction file list (to improve search speed)
all_pred_files = glob.glob(os.path.join(PRED_DIR, "*.npy")) + \
                 glob.glob("outputs_clear_*.npy") + \
                 glob.glob("privateST/results/outputs_clear_*.npy")

def find_file_by_sample(sid, file_list):
    """
    Finds the actual file (e.g., outputs_clear_..._22_33.npy)
    based on the sample_name in the CSV (e.g., C1_22_33).
    """
    parts = sid.split('_')
    if len(parts) < 2: return None

    # Extract coordinate part (last two numbers)
    # e.g., "Sample_22_33" -> "22", "33"
    target_coords = parts[-2:]

    for f in file_list:
        # Consider it a match if the filename contains all coordinate numbers
        if all(p in f for p in target_coords):
            return f
    return None

def main():
    try:
        print("🚀 [Start] Evaluation Setup")
        print(f"   -> True Source: {TRUE_NPZ_PATH}")
        print(f"   -> Sample List: {NAMES_CSV_PATH}")

        # ---------------------------------------------------------------------
        # 1. Load Ground Truth (True Value)
        # ---------------------------------------------------------------------
        if not os.path.exists(TRUE_NPZ_PATH):
            raise FileNotFoundError(f"Ground Truth file not found: {TRUE_NPZ_PATH}")

        z = np.load(TRUE_NPZ_PATH, allow_pickle=True)

        # Load user-specified True value
        # Note: scaled_y.npz preserves the original keys (likely 'counts' or 'true')
        y_true = z['counts'] if 'counts' in z else (z['true'] if 'true' in z else z['yt'])

        print(f"✅ True Data Loaded: Shape {y_true.shape}")
        print(f"   -> Stats: Mean={y_true.mean():.4f}, Max={y_true.max():.4f}")

        # ---------------------------------------------------------------------
        # 2. Load Predictions (Matched by CSV Order)
        # ---------------------------------------------------------------------
        if not os.path.exists(NAMES_CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {NAMES_CSV_PATH}")

        # Load CSV
        sample_names = pd.read_csv(NAMES_CSV_PATH)["sample_name"].astype(str).tolist()

        # Validate data counts
        n_csv = len(sample_names)
        n_npz = y_true.shape[0]

        if n_csv != n_npz:
            print(f"⚠️ Warning: Count mismatch! CSV({n_csv}) vs NPZ({n_npz})")
            print("   -> Trimming to the smaller count.")
            n_samples = min(n_csv, n_npz)
            sample_names = sample_names[:n_samples]
            y_true = y_true[:n_samples]
        else:
            n_samples = n_csv
            print(f"✅ Sample count matched: {n_samples}")

        print(f"🔍 Matching .npy files for {n_samples} samples...")

        pred_list = []
        missing_cnt = 0

        for i, sn in enumerate(sample_names):
            # 1. Find prediction file matching the sample name
            f_path = find_file_by_sample(sn, all_pred_files)

            # 2. Load and add to list
            if f_path:
                pv = np.load(f_path).flatten()
                pred_list.append(pv)
            else:
                # Fill with zeros if file not found (maintain dimension)
                pred_list.append(np.zeros(250))
                missing_cnt += 1
                if missing_cnt <= 3: print(f"   ⚠️ Missing prediction for: {sn}")

        y_pred = np.array(pred_list)
        print(f"✅ Prediction Data Constructed: Shape {y_pred.shape}")

        if missing_cnt > 0:
            print(f"   ⚠️ Total missing files: {missing_cnt}")

        # ---------------------------------------------------------------------
        # 3. Determine Ranking (Based on Mean of True Values)
        # ---------------------------------------------------------------------
        # Select Top 100 based on the mean of True values (Standardized or raw from NPZ)
        true_means = np.mean(y_true, axis=0)
        sort_idx = np.argsort(true_means)[::-1]
        print(f"✅ Ranking determined by NPZ True values.")

        # ---------------------------------------------------------------------
        # 4. Calculate PCC
        # ---------------------------------------------------------------------
        gene_pccs = []
        for j in range(y_true.shape[1]):
            t = y_true[:, j]
            p = y_pred[:, j]

            # Check standard deviation (prevent constant values)
            if np.std(t) > 1e-9 and np.std(p) > 1e-9:
                corr, _ = pearsonr(t, p)
                gene_pccs.append(corr)
            else:
                gene_pccs.append(0.0)

        # ---------------------------------------------------------------------
        # 5. Print Final Results
        # ---------------------------------------------------------------------
        # Top 10
        avg_pcc_10 = np.mean([gene_pccs[i] for i in sort_idx[:10]])
        # Top 100 (Target Metric)
        avg_pcc_100 = np.mean([gene_pccs[i] for i in sort_idx[:100]])
        # All 250
        avg_pcc_250 = np.mean(gene_pccs)

        print("\n" + "="*85)
        print("Final Gene-wise Evaluation Summary")
        print(f"(True: scaled_y.npz | Match: {os.path.basename(NAMES_CSV_PATH)})")
        print("-" * 85)
        print(f"{'Metric':<45} | {'Value':<10}")
        print("-" * 85)
        print(f"{'Mean PCC (Top 4% - 10 Genes)':<45} | {avg_pcc_10:.6f}")
        print(f"{'Mean PCC (Top 40% - 100 Genes)':<45} | {avg_pcc_100:.6f}")
        print(f"{'Mean PCC (All - 250 Genes)':<45} | {avg_pcc_250:.6f}")
        print("="*85 + "\n")

        # Save Results
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Gene_Index", "PCC"])
            for rank, idx in enumerate(sort_idx):
                writer.writerow([rank+1, idx, gene_pccs[idx]])
        print(f"Detailed results saved to: {CSV_OUT}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()                             
