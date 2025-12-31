import os
import numpy as np
import pickle
import csv
import re
from scipy.stats import pearsonr

try:
    # 1. Setup Paths
    root_dir = "test/counts/512/Breast_cancer/"
    gt_dir = os.path.join(root_dir, "BC23287/")
    pred_dir = "results/"
    gene_p = os.path.join(root_dir, "gene.pkl")
    mean_p = os.path.join(root_dir, "mean_expression.npy")
    csv_out = os.path.join(pred_dir, "gene_evaluation_results.csv")

    # 2. Replicate Spatial Class Filtering Logic
    with open(gene_p, "rb") as f:
        all_genes = pickle.load(f)
    mean_expression = np.load(mean_p)
    gene_pairs = sorted(zip(mean_expression, range(len(mean_expression))))
    top_250_set = set([p[1] for p in gene_pairs[::-1][:250]])
    target_indices = [i for i in range(len(all_genes)) if i in top_250_set]

    # 3. Match Data by Pixel Coordinates
    print("Matching files based on pixel coordinates...")

    # Map GT pixel coordinates {(x, y): filename}
    gt_pixel_map = {}
    for f in os.listdir(gt_dir):
        if f.endswith('.npz'):
            data = np.load(os.path.join(gt_dir, f))
            px, py = data['pixel']  # Expected format: [x, y]
            gt_pixel_map[(int(px), int(py))] = f

    # Map Pred pixel coordinates {(x, y): filename}
    pd_pixel_map = {}
    for f in os.listdir(pred_dir):
        if f.endswith('.npy'):
            # Extract numerical coordinates from filename
            # Example: 'ResNet_Approx_C1_2608_4623' -> (2608, 4623)
            nums = re.findall(r'\d+', f)
            if len(nums) >= 2:
                px, py = int(nums[-2]), int(nums[-1])
                pd_pixel_map[(px, py)] = f

    common_pixels = set(gt_pixel_map.keys()).intersection(set(pd_pixel_map.keys()))

    print(f"Total GT files: {len(gt_pixel_map)}")
    print(f"Total Pred files: {len(pd_pixel_map)}")
    print(f"Matched common files: {len(common_pixels)}")

    if not common_pixels:
        print("\nMatching failed. Please check if pixel values align.")
        print(f"GT Coordinate Sample: {list(gt_pixel_map.keys())[:2]}")
        print(f"Pred Coordinate Sample: {list(pd_pixel_map.keys())[:2]}")
        raise Exception("No matching coordinates found.")

    gt_mat, pd_mat = [], []
    for pix in sorted(list(common_pixels)):
        try:
            # Load Ground Truth
            go = np.load(os.path.join(gt_dir, gt_pixel_map[pix]))
            gv = go['count'].flatten()

            # Load Prediction
            pv = np.load(os.path.join(pred_dir, pd_pixel_map[pix])).flatten()

            # Apply filtering and normalization (log1p for GT)
            gt_mat.append(np.log1p(gv[target_indices]))
            pd_mat.append(pv)
        except Exception:
            continue

    gt_mat, pd_mat = np.array(gt_mat), np.array(pd_mat)

    # 4. Calculate Correlation
    gene_pccs = []
    for j in range(250):
        try:
            # Calculate Pearson Correlation Coefficient (PCC)
            pcc, _ = pearsonr(gt_mat[:, j], pd_mat[:, j])
            gene_pccs.append(pcc if not np.isnan(pcc) else 0.0)
        except Exception:
            gene_pccs.append(0.0)

    # --- Results Reporting ---
    header = "{:<5} | {:<25} | {:<25}".format("No.", "ENSG ID", "PCC (Pearson Correlation)")
    print("\n" + "="*85 + "\nGene-wise Evaluation (Top 250 Target Genes)\n" + "-"*85)
    print(header)
    print("-" * 85)

    csv_data = []
    for i in range(250):
        ensg = all_genes[target_indices[i]]
        # Handle byte strings if necessary
        ensg_id = (ensg.decode() if isinstance(ensg, bytes) else ensg).split('.')[0]
        pcc_val = gene_pccs[i]
        csv_data.append([i+1, ensg_id, pcc_val])

        # Print summary (first 10 and last 5 for brevity)
        if i < 10 or i >= 245:
            print(f"{i+1:<5} | {ensg_id:<25} | {pcc_val:.6f}")
        elif i == 10:
            print("...")

    avg_pcc = np.nanmean(gene_pccs)
    print("-" * 85 + f"\nFinal Mean Gene-wise PCC: {avg_pcc:.6f}")

    # 5. Export to CSV
    if not os.path.exists(os.path.dirname(csv_out)):
        os.makedirs(os.path.dirname(csv_out))

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ENSG_ID", "PCC"])
        writer.writerows(csv_data)

    print(f"\nEvaluation results saved to: {csv_out}\n" + "="*85)

except Exception as e:
    print(f"\nError: {e}")
