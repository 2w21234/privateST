# privateST: Privacy-Preserving Deep Learning Inference for Spatial Transcriptomics

This repository provides the code to perform privacy-preserving inference on a pre-trained ResNet18 model using homomorphic encryption. The model is trained using the **BrstNet** code to predict spatial transcriptomics data from histopathology images. The goal is to demonstrate a workflow where sensitive patient data (histopathology images) can be processed on an external server without exposing the raw information[cite: 40, 41, 42].

The trained model (`model/epoch_11_model_state_dict.pth`) is loaded, and inference is run on encrypted data using the **Orion** framework.

![Workflow of privateST](https://storage.googleapis.com/kms-space/privateST/figure1.png)

## 1. Data Description

The dataset used here is a scaled-down version of the public breast cancer spatial transcriptomics dataset, which originally contained data from 22 training patients and 1 test patient. Due to GitHub size limitations, this repository includes only a subset of the data for one training and one test patient.

* **Training Counts Root:** `./training/counts/512/Breast_cancer`
* **Training Images Root:** `./training/images/512/Breast_cancer`
* **Test Images Root:** `./test/images/64`
    * This directory contains the input images for `privateST`. The original 512x512x3 pixel images have been resized to 64x64x3 using bilinear interpolation.
* **Test Patients CSV:** `./test/test_patients.csv`
* **Test Counts Root:** `./test/counts/512/Breast_cancer`
* **Gene Filter:** Set to `250`, meaning the model predicts the expression for the top 250 genes with the highest mean expression.

---

## 2. Precomputed Statistics

The `./precomputed_stats/` folder contains files generated from the full training set that are necessary for data preprocessing:

* `gene.pkl`: A list of the gene names.
* `mean_expression.npy`: The corresponding mean gene expression values.
    * `gene.pkl` and `mean_expression.npy` are used to select the top 250 target genes.
* `image_stats.csv`: Contains the mean and standard deviation for each color channel (RGB) of the image patches. These values are used for image normalization.

---

## 3. Setup and Execution

Follow these steps to set up the environment and run the inference script.

**System Requirements:**
* A system with at least **512 GB of RAM** is recommended for stable operation.

**Installation Steps:**
1.  Create and activate the Conda environment using the provided file:
    ```bash
    conda env create -f environment.yml
    conda activate privateST
    ```

**Running Inference:**
1.  Run the test script to perform inference on the encrypted test data:
    ```bash
    python test_privateST.py
    ```
2.  The `test_privateST.sh` file is an example script used for running the job on a **SLURM** cluster.

---

## 4. Outputs
Running ```test_privateST.py``` creates a ```./results/``` folder where the following three types of inference result files are saved:

PyTorch model inference: The output from inference using the standard PyTorch model.

Orion (clear version): The output when calculated in the polynomial space without actual encryption.

Orion (FHE version): The final inference results on the actually encrypted data.

---

## 5. Implementation Notes

### Orion Compatibility

To ensure compatibility with the Orion framework, the standard `BasicBlock` from PyTorch's ResNet model has been redefined within the main script (`test_privateST.py`) as `CustomBasicBlock`. This custom implementation uses the same operations but conforms to the structure expected by Orion.

### Training a New ResNet Model

If you want to train a new ResNet18 model from scratch that is compatible with the Orion workflow, you must slightly modify the `torchvision` library's source files.

1.  Locate the `resnet.py` file in your Conda environment. It is typically found at:
    `/home/[user name]/.conda/envs/orion/lib/python3.13/site-packages/torchvision/models/resnet.py`
2.  Replace that file with the `resnet.py` file provided in this repository.
    * This modified file ensures that the saved model architecture after training perfectly matches the structure expected by the Orion inference code.
  
   
---
   
## 6. References
1. Orion: A Fully Homomorphic Encryption Framework for Deep Learning, March 2025, DOI: 10.1145/3676641.3716008
2. Breast cancer histopathology image-based gene expression prediction using spatial transcriptomics data and deep learning, May 2023, DOI: 10.1038/s41598-023-40219-0

