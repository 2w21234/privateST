# privateST: Privacy-Preserving Deep Learning Inference for Spatial Transcriptomics

This repository provides the code to perform privacy-preserving inference on a pre-trained ResNet18 model using homomorphic encryption. The ResNet18 model with avgerage Pooling is trained  to predict spatial transcriptomics data from histopathology images. The goal is to demonstrate a workflow where sensitive patient data (histopathology images) can be processed on an external server without exposing the raw information.

The trained model (`model/epoch_11_model_state_dict.pth`) is loaded, and inference is run on encrypted data using the **Orion** framework.

<img width="2457" height="2268" alt="image" src="https://github.com/user-attachments/assets/50cb3c10-2300-45c9-af2e-b0450fe8a733" />
![Uploading image.png…]()


## 0. Dependencies

This project was developed and tested with the specific library versions listed below. It is recommended to check your environment's versions to prevent compatibility issues.


* **System**: Linux 9.4

* **Conda**: 24.5.0

* **Python**: 3.10.6

* **PyTorch**: 2.5.1

* **Torchvision**: A version compatible with PyTorch


---


## 1. Data Description

The dataset provided here is a scaled-down version of the breast cancer spatial transcriptomics dataset.Test Set: The test set is identical to the one used for evaluation in the associated research paper.Test Images: Located in ./test/images/64. The original 512x512x3 pixel images were resized to 64x64x3 using bilinear interpolation. The test set is identical to the one evaluated in the associated research paper and consists of 529 samples in total.

* **Training Counts Root:** `./training/counts/224/Breast_cancer`
* **Training Images Root:** `./training/images/224/Breast_cancer`
* **Test Images Root:** `./test/images/224`
* **Test Patients CSV:** `./test/test_patients.csv`
* **Test Counts Root:** `./test/counts/224/Breast_cancer`

   The directory contains .npz files for each spot, where the count key stores the raw expression values.
* **Normalized True Expression:** `epoch_15.npz`
  
   epoch_15.npz contains both the Normalized True Expression and Predicted Values.
   The predictions were generated using a model architecture with Max Pooling for downsampling and Standard ReLU activation (actual ReLU, not a polynomial approximation).
* **Trained Model:** The model weights are provided in ```model/epoch_15_model_state_dict.pth```. This model was trained on the 22 training patients as described in the original paper.

---

## 2. Precomputed Statistics

The `./precomputed_stats/` folder contains files generated from the full training set that are necessary for data preprocessing:

* `gene.pkl`: A list of the gene names.
* `mean_expression.npy`: The corresponding mean gene expression values.
* `image_stats.csv`: Contains the mean and standard deviation for each color channel (RGB) of the image patches. These values are used for image normalization.

---

## 3. Setup and Execution

Follow these steps to set up the environment and run the inference script.

**Installation Steps:**
1.  Create and activate the Conda environment using the provided file:
    ```
    conda env create -f environment.yml
    conda activate privateST
    ```

**Running Inference with Options:**
Running Inference with Options
You can control the inference process using command-line arguments. This is particularly useful for quick verification or full encrypted evaluation


###    ⚠️ System Requirements & Memory Warning
The full **FHE (Homomorphic Encryption)** inference process is extremely memory-intensive.

* **RAM Requirement**: **Minimum 512GB RAM** is required for full FHE inference.
* **Important Warning**: If you run the script **without** the `--approx_only` flag on a system with less than 512GB of RAM, the process will fail with a `MemoryError` or system crash.
  
### Quick Verification (ResNet_Approx Only)
If your system does not meet the 512GB RAM requirement, you **must** use the `--approx_only` flag to run only the ResNet_Approx mode.

    
      # Run ONLY the ResNet_Approx mode and skip FHE inference
      python test_privateST.py --approx_only
    
--approx_only: The script will exit after saving the .npy files for the ResNet_Approx mode, skipping the time-consuming FHE compilation and encrypted inference.

### Full Inference (ResNet_Approx + ResNet_HE(privateST))
    
      # Run the full pipeline
      python test_privateST.py 
    


---

## 4. Outputs
Running ```test_privateST.py``` creates a ```./results/``` folder where the following three types of inference result files are saved:

PyTorch model inference: The output from inference using the standard PyTorch model.

Orion (Approx version): The output when calculated in the polynomial space without actual encryption. (Files are prefixed with Approx_)

Orion (HE version): The final inference results on the actually encrypted data. (Files are prefixed with HE_)

---


## 5. Evaluation
To evaluate the performance of the inference results, run the measure.py script.


```python measure.py```


Functionality: This script processes the result files saved in the ```./results/``` directory.

Metrics: It calculates the following performance metrics to compare the predicted expression levels with the ground truth:

Average PCC: Pearson Correlation Coefficient

Average SCC: Spearman Correlation Coefficient

Average RMSE: Root Mean Square Error


---


## 6. Implementation Notes

### Orion Compatibility

To ensure compatibility with the Orion framework, the standard `BasicBlock` from PyTorch's ResNet model has been redefined within the main script (`test_privateST.py`) as `CustomBasicBlock`. This custom implementation uses the same operations but conforms to the structure expected by Orion.

### Training a New ResNet Model
To ensure your model is fully compatible with the Orion (FHE) workflow, you must initialize it using the ```custom_resnet18``` function defined in ```test_privateST.py```. This ensures the architecture—specifically the pooling layers and residual blocks—perfectly matches the structure required for encrypted inference.

#### **Why use the Custom Initialization Function?**
Standard PyTorch/torchvision `ResNet18` models use **Max Pooling** by default. However, for efficient and accurate Homomorphic Encryption (HE) inference, the **privateST** pipeline requires **Average Pooling**. 

Using the provided function instead of the standard torchvision initialization correctly configures the internal naming and operations necessary for the Orion workflow.
Using the Custom Initialization Function
Instead of using the standard torchvision initialization, use the provided function to build the model. This correctly configures the internal naming and operations (such as Average Pooling) necessary for the privateST pipeline.
```
# Import the custom initialization function from the project script
from test_privateST import custom_resnet18

# Initialize the model for your specific task 
model = custom_resnet18(num_classes=250)

# The function automatically configures:
# - AvgPool2d (mapped to self.pool) for HE efficiency
# - CustomBasicBlock with F.interpolate logic for skip-connection stability
```
By following this approach, the saved checkpoints (.pth) will align with the Orion inference engine without the need for manual structural adjustments or weight remapping.

---
   
## 6. References
1. Orion: A Fully Homomorphic Encryption Framework for Deep Learning, March 2025, DOI: 10.1145/3676641.3716008
2. Breast cancer histopathology image-based gene expression prediction using spatial transcriptomics data and deep learning, May 2023, DOI: 10.1038/s41598-023-40219-0

