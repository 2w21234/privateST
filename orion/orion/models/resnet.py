import torch.nn as nn
import orion.nn as on
print('resnet loaded!!!!')

class BasicBlock(on.Module):
    expansion = 1

    def __init__(self, Ci, Co, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(Ci, Co, kernel_size=3, stride=stride, padding=1, bias=False)
        #print('Block Conv1 :', out) 
        self.bn1   = on.BatchNorm2d(Co)
        #print('Block bn1 :', out)
        self.act1  = on.ReLU()
        #print('Block act1 :', out)

        self.conv2 = on.Conv2d(Co, Co, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = on.BatchNorm2d(Co)
        #print('Block Conv2 :', out)
        self.act2  = on.ReLU()
        #print('Block bn2 :', out)
       
        self.add = on.Add()
        self.shortcut = nn.Sequential()
        if stride != 1 or Ci != self.expansion*Co:
            self.shortcut = nn.Sequential(
                on.Conv2d(Ci, self.expansion*Co, kernel_size=1, stride=stride, bias=False),
                on.BatchNorm2d(self.expansion*Co))
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        return self.act2(out)
    
    """
    def forward(self, x):
        out = self.conv1(x)
        #print('Block Conv1 :', out)
        out = self.bn1(out)
        #print('Block bn1 :', out)
        out = self.act1(out)
        #print('Block act1 :', out)
        #print('✅ act1 출력 shape:', out.shape)
        out = self.bn2(self.conv2(out))
        #print('Block bn2 :', out)
        out = self.add(out, self.shortcut(x))
        #print('Added',out)
        #print('Block act2 :',self.act2(out))
        return self.act2(out)
    """
    

class Bottleneck(on.Module):
    expansion = 4

    def __init__(self, Ci, Co, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(Ci, Co, kernel_size=1, bias=False)
        self.bn1   = on.BatchNorm2d(Co)
        self.act1  = on.SiLU(degree=127) 

        self.conv2 = on.Conv2d(Co, Co, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = on.BatchNorm2d(Co)
        self.act2  = on.SiLU(degree=127)  

        self.conv3 = on.Conv2d(Co, Co*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3   = on.BatchNorm2d(Co*self.expansion)
        self.act3  = on.SiLU(degree=127)  

        self.add = on.Add()
        self.shortcut = nn.Sequential()
        if stride != 1 or Ci != self.expansion*Co:
            self.shortcut = nn.Sequential(
                on.Conv2d(Ci, self.expansion*Co, kernel_size=1, stride=stride, bias=False),
                on.BatchNorm2d(self.expansion*Co))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.add(out, self.shortcut(x))
        return self.act3(out)
    

class ResNet(on.Module):
    def __init__(self, dataset, block, num_blocks, num_chans, conv1_params, num_classes):
        super().__init__()
        self.in_chans = num_chans[0]
        self.last_chans = num_chans[-1]

        self.conv1 = on.Conv2d(3, self.in_chans, **conv1_params, bias=False)
        self.bn1 = on.BatchNorm2d(self.in_chans)
        self.act = on.ReLU()
         
        #self.pool = nn.Identity() 
        
        self.First_Avgpool = on.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        
        #if dataset == 'imagenet': # for imagenet we must also downsample
        #    self.pool = on.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            stride = 1 if i == 0 else 2
            self.layers.append(self.layer(block, num_chans[i], num_blocks[i], stride))

        self.avgpool = on.AdaptiveAvgPool2d(output_size=(1,1)) 
        self.flatten = on.Flatten()
        self.linear  = on.Linear(self.last_chans * block.expansion, num_classes)

    def layer(self, block, chans, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_chans, chans, stride))
            self.in_chans = chans * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #print("[conv1] output:\n", out)

        out = self.bn1(out)
        #print("[bn1] output:\n", out)

        out = self.act(out)
        #print("[relu] output:\n", out)
 
        out = self.First_Avgpool(out)
        #print("[First avg pool] output:\n", out)
         
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            #print(f"[layer{idx+1}] output:\n", out)

        out = self.avgpool(out)
        #print("[avgpool] output:\n", out)

        out = self.flatten(out)
        #print("[flatten] output:\n", out)

        out = self.linear(out)
        #print("[linear] output:\n", out)

        return out

    """ 
    def forward(self, x): 
        out = self.act(self.bn1(self.conv1(x)))
        #print('1: ', out[0,0,:4,:4])
        #out = self.pool(out)
        #print('2: ', out[0,0,:4,:4])
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)
    """

################################
# CIFAR-10 / CIFAR-100 ResNets #
################################

def ResNet20(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [3,3,3], [16,32,64], conv1_params, num_classes)

def ResNet32(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [5,5,5], [16,32,64], conv1_params, num_classes)

def ResNet44(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [7,7,7], [16,32,64], conv1_params, num_classes)

def ResNet56(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [9,9,9], [16,32,64], conv1_params, num_classes)

def ResNet110(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [18,18,18], [16,32,64], conv1_params, num_classes)

def ResNet1202(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [200,200,200], [16,32,64], conv1_params, num_classes)

####################################
# Tiny ImageNet / ImageNet ResNets #
####################################

def ResNet18(dataset='brstnet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [2,2,2,2], [64,128,256,512], conv1_params, num_classes)

def ResNet34(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [3,4,6,3], [64,128,256,512], conv1_params, num_classes)

def ResNet50(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,4,6,3], [64,128,256,512], conv1_params, num_classes)

def ResNet101(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,4,23,3], [64,128,256,512], conv1_params, num_classes)

def ResNet152(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,8,36,3], [64,128,256,512], conv1_params, num_classes)


def get_resnet_config(dataset):
    configs = {
        "cifar10": {"kernel_size": 3, "stride": 1, "padding": 1, "num_classes": 10},
        "cifar100": {"kernel_size": 3, "stride": 1, "padding": 1, "num_classes": 100},
        "tiny": {"kernel_size": 7, "stride": 1, "padding": 3, "num_classes": 200},
        "imagenet": {"kernel_size": 7, "stride": 2, "padding": 3, "num_classes": 1000},
        "brstnet": {"kernel_size": 7, "stride": 2, "padding": 3, "num_classes": 250},
    }

    if dataset not in configs:
        raise ValueError(f"ResNet with dataset {dataset} is not supported.")
    
    config = configs[dataset]
    conv1_params = {
        'kernel_size': config["kernel_size"],
        'stride': config["stride"],
        'padding': config["padding"]
    }
    
    return conv1_params, config["num_classes"]


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = ResNet50()
    net.eval()

    x = torch.randn(1,3,224,224)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,224,224), depth=10)
    print("Total flops: ", total_flops)
