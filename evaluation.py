import os; import numpy as np; import pickle; import csv; from scipy.stats import pearsonr
try:
    root_dir = "test/counts/512/Breast_cancer/"
    gt_dir, pred_dir = os.path.join(root_dir, "BC23287/"), "results/"
    gene_p, mean_p = os.path.join(root_dir, "gene.pkl"), os.path.join(root_dir, "mean_expression.npy")
    with open(gene_p, "rb") as f: all_genes = pickle.load(f)
    mean_expression = np.load(mean_p)
    gene_pairs = sorted(zip(mean_expression, range(len(mean_expression))))
    top_250_indices_set = set([p[1] for p in gene_pairs[::-1][:250]])
    target_indices = [i for i in range(len(all_genes)) if i in top_250_indices_set]
    gt_fs = {f.replace('.npz',''): f for f in os.listdir(gt_dir) if f.endswith('.npz')}
    pd_fs = {f.replace('.npy',''): f for f in os.listdir(pred_dir) if f.endswith('.npy')}
    common = sorted(list(set(gt_fs.keys()).intersection(set(pd_fs.keys()))))
    gt_mat, pd_mat = [], []
    for k in common:
        try:
            go = np.load(os.path.join(gt_dir, gt_fs[k]))
            pv = np.load(os.path.join(pred_dir, pd_fs[k])).flatten()
            gv = (go['count'] if 'count' in go.files else go[go.files[0]]).flatten()
            gt_mat.append(np.log1p(gv[target_indices])); pd_mat.append(pv)
        except: continue
    gt_mat, pd_mat = np.array(gt_mat), np.array(pd_mat)
    gene_pccs = [pearsonr(gt_mat[:, j], pd_mat[:, j])[0] for j in range(250)]
    print("\n" + "="*85 + "\n📊 Full Gene-wise Evaluation (250 Target Genes)\n" + "-"*85)
    print(f"{'No.':<5} | {'ENSG ID':<25} | {'PCC (Pearson Correlation)':<25}")
    print("-" * 85)
    csv_data = []
    for i in range(250):
        ensg = all_genes[target_indices[i]]
        ensg_s = (ensg.decode() if isinstance(ensg, bytes) else ensg).split('.')[0]
        pcc_val = gene_pccs[i]
        csv_data.append([i, ensg_s, pcc_val])
        print(f"{i+1:<5} | {ensg_s:<25} | {pcc_val:.6f}")
    print("-" * 85)
    avg_pcc = np.nanmean(gene_pccs)
    print(f"✅ Final Mean Gene-wise PCC: {avg_pcc:.6f}")
    with open("gene_evaluation_results.csv", "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["Index", "ENSG_ID", "PCC"]); writer.writerows(csv_data)
    print(f"💾 Results successfully saved to 'gene_evaluation_results.csv'\n" + "="*85)
except Exception as e: print(f"\n❌ Error: {e}")
