import gc
import os
import sys
#sys.path.insert(0, './orion')
#sys.path.append('.')
# 스크립트 위치 기준
#script_dir = os.path.dirname(os.path.abspath(__file__))  # /home/2w21234/privateST
#orion_path = os.path.join(script_dir, 'orion')           # /home/2w21234/privateST/orion
#sys.path.insert(0, orion_path)

import time
import math
import psutil
import glob
import pickle
import random
import pathlib
import argparse
import collections
import torch
import numpy as np
sys.path.insert(1, './orion')
import orion
from orion.models.resnet import ResNet18
import inspect
from tqdm import tqdm
import sys
import os
import importlib
import pandas as pd
from torchvision import transforms
from orion.core.utils import get_tiny_datasets, mae
import utils
import ensembl_identity as ensembl
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32
from torchsummary import summary
import os, torch, numpy as np, torchvision
from PIL import Image
from tqdm import tqdm
import random
from torchvision.models.resnet import ResNet
import torchvision
import torch
import utils
import torch.nn.functional as F
import efficientnet_pytorch
import pytorch_pretrained_vit
import torch.nn as nn



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_ckpt         = "./model/epoch_11_model_state_dict.pth"
train_count_root    = "./training/counts/512/Breast_cancer"
train_img_root      = "./training/images/512/Breast_cancer"

test_patients_csv  = "./test/test_patients.csv"
test_count_root    = "./test/counts/512/Breast_cancer"
test_img_root      = "./test/images/64"
gene_filter        = 250
orion_batch_size   = 1
resol              = 64
device             = torch.device("cpu")  
precomputed_root   = "./precomputed_stats"
save_dir = "./results"


patients_df = pd.read_csv(test_patients_csv, header=None)
test_patients = patients_df.iloc[:, 0].tolist()  
img_stats_path = precomputed_root+"/image_gene_stats.csv"

stats_df = pd.read_csv(img_stats_path)

image_mean = stats_df[stats_df["type"] == "image_mean"]["value"].tolist()
image_std  = stats_df[stats_df["type"] == "image_std"]["value"].tolist()


class CustomBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.identity = nn.Identity()
        self.act2 = nn.ReLU(inplace=True) 

        if downsample is None and (stride != 1 or inplanes != planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False),
                norm_layer(planes)
            )
        else:
            self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if identity.shape != out.shape:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)

        out = self.identity(out + identity)
        out = self.act2(out)
        return out



def custom_resnet18(num_classes=10):
    model = ResNet(block=CustomBasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    return model

# ───── model loading ─────
model = custom_resnet18(num_classes=gene_filter)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.to(device)
model.eval()





state_dict = torch.load(model_ckpt, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}



def convert_pytorch_keys_to_orion(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("downsample", "shortcut") 
        if k.startswith("layer"):
            parts = k.split(".")
            layer_num = int(parts[0][5:])
            block_num = int(parts[1])
            rest = ".".join(parts[2:])
            new_key = f"layers.{layer_num - 1}.{block_num}.{rest}"
            new_state_dict[new_key] = v
        elif k.startswith("fc."):
            new_state_dict[k.replace("fc", "linear")] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

#print(state_dict)
orion_state = convert_pytorch_keys_to_orion(state_dict)
he_model = ResNet18(dataset='brstnet')
he_model.load_state_dict(orion_state, strict=False)
he_model.eval()

print("Orion model loaded.")





class Spatial(torch.utils.data.Dataset):
    def __init__(self,
                 patient=None,
                 count_root=None,
                 img_root=None,
                 gene_filter=gene_filter,
                 aux_ratio=0,
                 transform=None,
                 normalization=None,
                 gene_info_root=None):
        self.dataset=glob.glob(os.path.join(count_root, "*", "*.npz"))
                
        
        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

        self.transform = transform
        self.count_root = count_root
        self.img_root = img_root
        self.gene_filter = gene_filter
        self.aux_ratio = aux_ratio
        self.normalization = normalization
        self.count_root = count_root
        subtype_files = [os.path.join(self.count_root, "subtype.pkl"), os.path.join(self.count_root, "subtype.pkl")]
        self.subtype = {}
        self.gene_info_root = gene_info_root
        for file in subtype_files:
            try:
                with open(file, "rb") as f:
                    self.subtype.update(pickle.load(f))
            except FileNotFoundError:
                print(f"⚠ Warning: {file} not found.")

        print(f"✅ Loaded {len(self.subtype)} subtype entries.")

        gene_pkl_path = os.path.join(self.gene_info_root, "gene.pkl")
        mean_expression_path = os.path.join(self.gene_info_root, "mean_expression.npy")
        
        print('gene_pkl_path : ',gene_pkl_path)
        print('mean_expression_path :',mean_expression_path)
        with open(gene_pkl_path, "rb") as f:
            self.ensg_names = pickle.load(f)
        self.mean_expression = np.load(mean_expression_path)

        self.gene_names = list(map(lambda x: ensembl.symbol[x], self.ensg_names))

        keep_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:self.gene_filter]))[1])

        self.keep_bool = np.array([i in keep_gene for i in range(len(self.gene_names))])

        self.ensg_keep = [n for (n, f) in zip(self.ensg_names, self.keep_bool) if f]
        self.gene_keep = [n for (n, f) in zip(self.gene_names, self.keep_bool) if f]

        if self.aux_ratio != 0:
            self.aux_nums = int((len(self.gene_names) - self.gene_filter) * self.aux_ratio)
            aux_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][self.gene_filter:self.gene_filter + self.aux_nums]))[1])
            self.aux_bool = np.array([i in aux_gene for i in range(len(self.gene_names))])
            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, self.aux_bool) if f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        npz = np.load(self.dataset[index])
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]
        pt_filename = f"input_tensor_{section}_{coord[0]:04d}_{coord[1]:04d}.pt"
        pt_path = os.path.join(self.img_root, pt_filename)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"No PT file: {pt_path}")
        X = torch.load(pt_path)
        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        keep_count = count[self.keep_bool]
        y = torch.as_tensor(keep_count, dtype=torch.float)
        y = torch.log(1 + y)

        if self.normalization is not None:
            y = (y - self.normalization[0]) / self.normalization[1]

        if self.aux_ratio != 0:
            aux_count = count[self.aux_bool]
            aux = torch.as_tensor(aux_count, dtype=torch.float)
            aux = torch.log(1 + aux)

            return X, y, aux, coord, index, patient, section, pixel
        else:
            return X, y, coord, index, patient, section, pixel








class Spatial_train(torch.utils.data.Dataset):
    def __init__(self,
                 patient=None,
                 #window=64,
                 resolution=64,
                 count_root=None,
                 img_root=None,
                 gene_filter=250,
                 aux_ratio=0,
                 transform=None,
                 normalization=None,
                 gene_info_root=None,
                 ):
        self.dataset = sorted(glob.glob(f"{count_root}/*/*.npz"))
        if patient is not None:
           self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
        self.transform = transform
        #self.window = window
        self.resolution = resolution
        self.count_root = count_root
        self.img_root = img_root
        self.gene_filter = gene_filter
        self.aux_ratio = aux_ratio
        self.normalization = normalization
        self.gene_info_root=gene_info_root

        self.gene_info_path = self.gene_info_root if self.gene_info_root is not None else self.count_root
        self.gene_pkl_path = os.path.join(self.gene_info_path, "gene.pkl")
        self.mean_expression_path = os.path.join(self.gene_info_path, "mean_expression.npy")
        subtype_files = [os.path.join(self.count_root, "subtype.pkl")]
        self.subtype = {}
        
        for file in subtype_files:
            try:
                with open(file, "rb") as f:
                    self.subtype.update(pickle.load(f))
            except FileNotFoundError:
                print(f"⚠ Warning: {file} not found.")

        print(f"✅ Loaded {len(self.subtype)} subtype entries.")
             
        print('gene_pkl_path : ',self.gene_pkl_path)
        print('mean_expression_path :',self.mean_expression_path)

        with open(self.gene_pkl_path, "rb") as f:
            self.ensg_names = pickle.load(f)
        self.mean_expression = np.load(self.mean_expression_path)
        self.gene_names = list(map(lambda x: ensembl.symbol[x], self.ensg_names))

        keep_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:self.gene_filter]))[1])
        self.keep_bool = np.array([i in keep_gene for i in range(len(self.gene_names))])
        self.ensg_keep = [n for (n, keep) in zip(self.ensg_names, self.keep_bool) if keep]
        self.gene_keep = [n for (n, keep) in zip(self.gene_names, self.keep_bool) if keep]

        print(f"✅ Top250 ∩ Available genes ({len(self.gene_keep)}):", self.gene_keep)
        

        if self.aux_ratio != 0:
            self.aux_nums = int((len(self.gene_names) - self.gene_filter) * self.aux_ratio)
            aux_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][self.gene_filter:self.gene_filter + self.aux_nums]))[1])
            self.aux_bool = np.array([i in aux_gene for i in range(len(self.gene_names))])
            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, self.aux_bool) if f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        npz = np.load(self.dataset[index])
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        cached_image = os.path.join(
            self.img_root,
            patient,
            f"{section}_{coord[0]}_{coord[1]}.jpg"
        )

        X = Image.open(cached_image).convert("RGB")

        if self.transform is not None:
            X = self.transform(X)
        # Bilinear interpolation 
        if X.shape[1] != self.resolution:
            X = torchvision.transforms.Resize((self.resolution, self.resolution))(X)
            #print('Resized to 64x64')
        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        keep_count = count[self.keep_bool]
        y = torch.as_tensor(keep_count, dtype=torch.float)
        y = torch.log(1 + y)
        genes_bool = self.keep_bool
        genes = self.gene_keep
        ensg = self.ensg_keep
        
        if self.normalization is not None:
            y = (y - self.normalization[0]) / self.normalization[1]
                    
        if self.aux_ratio != 0:
            print('Aux ratio is not zero!')
            aux_count = count[self.aux_bool]
            aux = torch.as_tensor(aux_count, dtype=torch.float)
            aux = torch.log(1 + aux)

            return X, y, aux, coord, index, patient, section, pixel, genes_bool, genes, ensg
        else:
            return X, y, coord, index, patient, section, pixel, genes_bool, genes, ensg



# ───── DataLoader ─────
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std)
])


def get_spatial_patients(img_root):
    print(f"Scanning directory: {img_root}")
    patient_section = map(
        lambda x: (x.split("/")[-2], x.split("/")[-1].split("_")[0]),
        glob.glob(f"{img_root}/*/")  
    )
    print(f"Patients found at: {img_root}/*/")

    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        if s not in patient[p]:
            patient[p].append(s)

    return patient



train_patients = sorted(get_spatial_patients(train_img_root).keys())

print('test:',test_patients)
print('train:', train_patients)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed) 


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=image_mean, std=image_std)])


train_dataset = Spatial_train(train_patients,
                        count_root=train_count_root,
                        img_root= train_img_root,
                        resolution=resol,
                        gene_filter=gene_filter,
                        aux_ratio=0,
                        transform=train_transform,
                        normalization=None,
                        gene_info_root=precomputed_root
                        )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=orion_batch_size,
                                           num_workers=2,
                                           shuffle=True,
                                           worker_init_fn=seed_worker, 
                                           generator=g)

test_dataset = Spatial(
    patient=test_patients,#patient_list,
    count_root=test_count_root, 
    img_root=test_img_root,
    gene_filter=gene_filter,
    aux_ratio=0,
    transform=test_transform, 
    normalization=None,
    gene_info_root=precomputed_root
    )



test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=orion_batch_size,
    num_workers=2,
    shuffle=False  
)



# ───── Hook ─────
activation = {}
def hooker(name):
    def fn(_, __, output):
        activation.setdefault(name, []).append(output)
    return fn

layers = {
    "conv1": he_model.conv1,
    "bn1":   he_model.bn1,
    "layer1":he_model.layers[0],
    "layer2":he_model.layers[1],
    "layer3":he_model.layers[2],
    "layer4":he_model.layers[3],
    "fc":    he_model.linear
}
for n, l in layers.items():
    l.register_forward_hook(hooker(n))

hook_handles = []
for n, l in layers.items():
    handle = l.register_forward_hook(hooker(n))
    hook_handles.append(handle)

for h in hook_handles:
    h.remove()



# Data root
test_count_root_1 = test_count_root+"/BC23287"
os.makedirs(save_dir, exist_ok=True)

npz_list = [f for f in os.listdir(test_count_root_1) if f.endswith(".npz")]




start = time.time()
# Orion cheme initialization
print("===== 1) Pytorch & orion (clear mode) inference starts =====")
scheme = orion.init_scheme("/home/2w21234/privateST/orion/configs/resnet.yml")

orion.fit(he_model, train_loader)


he_model.eval()
model.eval()

for i, sample in enumerate(test_loader):
    if len(sample) == 8:  
        X, y, aux, coord, index, patient, section, pixel = sample
    else:  
        X, y, coord, index, patient, section, pixel = sample

    sample_input = X  
    pixel = pixel.squeeze().tolist()
    x, y_ = pixel
    section = section[0]
    
    filename_clear = f"{save_dir}/outputs_clear_{section}_{x:04d}_{y_:04d}.npy"
    filename_pytorch   = f"{save_dir}/outputs_pytorch_{section}_{x:04d}_{y_:04d}.npy"
    if os.path.exists(filename_clear):
        print(f"The file already exists : {filename_clear} → pass")
        continue

    # Orion clear inference
    he_model.eval()
    out_clear = he_model(sample_input).detach().cpu().numpy()
    np.save(filename_clear, out_clear)
    print(f"[CLEAR] Saved: {filename_clear}")

    # pytorch inference
    model.eval()
    with torch.no_grad(): 
        out_pytorch = model(sample_input)
    np.save(filename_pytorch, out_pytorch.detach().cpu().numpy())
    print(f"[CLEAR] Saved: {filename_pytorch}")



sample_times = []
total_start_time = time.perf_counter()

print("===== 2) Orion (FHE) model compiling starts =====")
input_level = orion.compile(he_model)

print("===== 3) Orion (FHE) model compiling finished and fhe inference starts =====")
he_model.he()

for i, sample in enumerate(test_loader):
    if len(sample) == 8:  
        X, y, aux, coord, index, patient, section, pixel = sample
    else:  
        X, y, coord, index, patient, section, pixel = sample

    sample_input = X  
    print('test set input size : ', sample_input.shape)
    pixel = pixel.squeeze().tolist()
    #print(pixel)
    x, y_ = pixel
    section = section[0]
    filename_fhe   = f"{save_dir}/outputs_fhe_{section}_{x:04d}_{y_:04d}.npy" 
    vec_ptxt = orion.encode(sample_input, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    sample_start_time = time.perf_counter()    
    out_ctxt = he_model(vec_ctxt)
    sample_end_time = time.perf_counter()
    sample_duration = sample_end_time - sample_start_time
    sample_times.append(sample_duration)
        
    out_fhe = out_ctxt.decrypt().decode()
    out_fhe = np.array(out_fhe)
    np.save(filename_fhe, out_fhe)
    
    ## break
    
    # Memory clearance 
    del vec_ctxt, vec_ptxt, out_ctxt, out_fhe
    gc.collect() 

he_model.eval()
print("\n" + "="*50)
print("                 Inference times              ")
print("="*50)
print(f"Number of samples : {len(sample_times)}")
print('Inference times :', sample_times)
print("="*50)
