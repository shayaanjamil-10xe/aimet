# %% [markdown]
# # AutoQuant
# 
# This notebook contains an example of how to use AIMET AutoQuant feature.
# 
# AIMET offers a suite of neural network post-training quantization (PTQ) techniques that can be applied in succession. However, finding the right sequence of techniques to apply is time-consuming and can be challenging for non-expert users. We instead recommend AutoQuant to save time and effort.
# 
# AutoQuant is an API that analyzes the model and automatically applies various PTQ techniques based on best-practices heuristics. You specify a tolerable accuracy drop, and AutoQuant applies PTQ techniques cumulatively until the target accuracy is satisfied.
# 
# ## Overall flow
# 
# This example performs the following steps:
# 
# 1. Define constants and helper functions
# 2. Load a pretrained FP32 model
# 3. Run AutoQuant
# 
# <div class="alert alert-info">
# 
# Note
# 
# This notebook does not show state-of-the-art results. For example, it uses a relatively quantization-friendly model (Resnet18). Also, some optimization parameters like number of fine-tuning epochs are chosen to improve execution speed in the notebook.
# 
# </div>

# %%
import json
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [l.strip() for l in lines]

def read_annot_file(file_path):
    annots = read_text_file(file_path)
    annots = {l.split(" ")[0]: int(l.split(" ")[1]) for l in annots}
    return annots

from torchvision import transforms
ANNOT_FILE = "/datasets/imagenet1k/tags.txt"
DATASET_DIR = '/datasets/imagenet1k/new_images'
image_size = 224
images_mean = [0.485, 0.456, 0.406]
images_std  = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=images_mean,
                                std=images_std)
transforms = transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])
annots = read_annot_file(ANNOT_FILE)

import os
from glob import glob
import torch
from PIL import Image

class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, annots_dict, transform=None):
        self.dataset_dir = dataset_dir
        self.image_paths = glob(os.path.join(dataset_dir, '*.JPEG'))
        self.transform = transform
        self.annots_dict = annots_dict
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # print(f"{image_path=} \t {idx=}")
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image) if self.transform else image
        label = self.annots_dict[os.path.basename(image_path)]
        return image, label
    
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None, len=None):
        self.dataset_dir = dataset_dir
        self.image_paths = glob(os.path.join(dataset_dir, '*.JPEG'))
        if len:
            self.image_paths = self.image_paths[:len]
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # print(f"{image_path=} \t {idx=}")
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image) if self.transform else image
        return image
    
dataset = ImagenetDataset(DATASET_DIR, annots, transform=transforms)
dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)

unlabeled_dataset = UnlabeledDataset(DATASET_DIR, transform=transforms)
unlabeled_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)


# %% [markdown]
# ## 1. Define Constants and Helper functions
# 
# This section defines the following constants and helper functions:
# 
# - **EVAL_DATASET_SIZE** A typical value is 5000. In this example, the value has been set to 500 for faster execution.
# - **CALIBRATION_DATASET_SIZE** A typical value is 2000. In this example, the value has been set to 200 for faster execution.
# - **_create_sampled_data_loader()** returns a DataLoader based on the dataset and the number of samples provided.
# - **eval_callback()** defines an evaluation function for the model.

# %%

import torch.nn as nn
from tqdm import tqdm

import torch.utils

def accuracy_helper(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def eval_callback(model: nn.Module, iterations: int = 100, use_cuda: bool = False) -> float:
    """
    Evaluate the specified model using the specified number of samples batches from the
    validation set.
    :param model: The model to be evaluated.
    :param iterations: The number of batches to use from the validation set.
    :param use_cuda: If True then use a GPU for inference.
    :return: The accuracy for the sample with the maximum accuracy.
    """
    data_loader = dataloader
    device = torch.device('cpu')
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('use_cuda is selected but no cuda device found.')
            raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    if iterations is None:
        print('No value of iteration is provided, running evaluation on complete dataset.')
        # iterations = len(data_loader)
        iterations = 5000
    if iterations <= 0:
        print('Cannot evaluate on %d iterations', iterations)

    acc_top1 = 0
    acc_top5 = 0

    print("Evaluating nn.Module for %d iterations with batch_size %d",
                iterations, data_loader.batch_size)

    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        for i, (input_data, target_data) in tqdm(enumerate(data_loader), total=iterations):
            if i == iterations:
                break
            inputs_batch = input_data.to(device)
            target_batch = target_data.to(device)

            predicted_batch = model(inputs_batch)

            batch_avg_top_1_5 = accuracy_helper(output=predicted_batch, target=target_batch,
                                            topk=(1, 5))

            acc_top1 += batch_avg_top_1_5[0].item()
            acc_top5 += batch_avg_top_1_5[1].item()

    acc_top1 /= iterations
    acc_top5 /= iterations

    print(f"Avg accuracy Top 1: {acc_top1}%\nAvg accuracy Top 5: {acc_top5}%\non validation Dataset")

    return acc_top1

def pass_calibration_data(sim_model, use_cuda=False):
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)
    batch_size = dataloader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    idx = 0
    with torch.no_grad():
        for input_data, target_data in tqdm(dataloader):
            # if "cf135f199d8c7a9d0dce9aa35acfb4c70c14e0aa" not in path[0]:
            #     continue
            # if "cf" in path[0]:
            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)
            batch_cntr += 1
            if batch_cntr * batch_size >= samples:
                break

# %% [markdown]
# ## 2. Load a pretrained FP32 model
# 
# **Load a pretrained resnet18 model from torchvision.** 
# 
# You can load any pretrained PyTorch model instead.

# %%
from torchvision.models import resnet18

model = resnet18(pretrained=True).eval()

if torch.cuda.is_available():
    model.to(torch.device('cuda'))

accuracy = eval_callback(model)
print(f'- FP32 accuracy: {accuracy}')

# %% [markdown]
# ## 3. Run AutoQuant
# 
# **3.1 Create an AutoQuant object.**
# 
# The AutoQuant feature uses an unlabeled dataset to quantize the model. The **UnlabeledDatasetWrapper** class creates an unlabeled Dataset object from a labeled Dataset. 

# %%
from aimet_torch.auto_quant import AutoQuant
from torch.utils.data import Dataset

dummy_input = torch.randn((1, 3, 224, 224)).to(torch.device('cpu'))

auto_quant = AutoQuant(model,
                        dummy_input=dummy_input,
                        data_loader=unlabeled_dataloader,
                        eval_callback=eval_callback,
                        custom_forward_pass_callback=pass_calibration_data,
                        config_file="/home/shayan/Desktop/temp/aimet/my_config.json",
                        param_bw=4,
                        output_bw=4)

# %% [markdown]
# **3.2 Run AutoQuant inference**.
# 
# AutoQuant inference uses the **eval_callback** with the generic quantized model without applying PTQ techniques. This provides a baseline evaluation score before running AutoQuant optimization.

# %%
sim, initial_accuracy = auto_quant.run_inference()
print(f"- Quantized Accuracy (before optimization): {initial_accuracy}")

# %% [markdown]
# **3.3 Set AdaRound Parameters (optional)**.
# 
# AutoQuant uses predefined default parameters for AdaRound.
# These values were determined empirically and work well with the common models.
# 
# If necessary, you can use custom parameters for Adaround.
# This example uses very small AdaRound parameters for faster execution.

# %%
from aimet_torch.adaround.adaround_weight import AdaroundParameters

ADAROUND_DATASET_SIZE = 200
adaround_data_set = UnlabeledDataset(DATASET_DIR, transform=transforms, len=ADAROUND_DATASET_SIZE)
adaround_data_loader = torch.utils.data.DataLoader(adaround_data_set, batch_size=1, shuffle=False)
adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader), default_num_iterations=200)
auto_quant.set_adaround_params(adaround_params)

# %% [markdown]
# **3.4 Run AutoQuant Optimization**.
# 
# This step runs AutoQuant optimization. AutoQuant returns the following:
# - The best possible quantized model
# - The corresponding evaluation score
# - The path to the encoding file
# 
# The **allowed_accuracy_drop** indicates the tolerable accuracy drop. AutoQuant applies a series of quantization features until the target accuracy (FP32 accuracy - allowed accuracy drop) is satisfied. When the target accuracy is reached, AutoQuant returns immediately without applying furhter PTQ techniques. See the [AutoQuant User Guide](https://quic.github.io/aimet-pages/releases/latest/user_guide/auto_quant.html) and [AutoQuant API documentation](https://quic.github.io/aimet-pages/releases/latest/api_docs/torch_auto_quant.html) for details.

# %%
model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=40)
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy}")

# %% [markdown]
# ---
# 
# ## Next steps
# 
# The next step is to export this model for installation on the target.
# 
# **Export the model and encodings.**
# 
# - Export the model with the updated weights but without the fake quant ops. 
# - Export the encodings (scale and offset quantization parameters). AIMET QuantizationSimModel provides an export API for this purpose.
# 
# The following code performs these exports.

# %%
os.makedirs('./output/', exist_ok=True)
dummy_input = dummy_input.cpu()
sim.export(path='./output/', filename_prefix='resnet18_after_cle_bc', dummy_input=dummy_input)

# %% [markdown]
# ## For more information
# 
# See the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) for details about the AIMET APIs and optional parameters.
# 
# See the [other example notebooks](https://github.com/quic/aimet/tree/develop/Examples/torch/quantization) to learn how to use other AIMET post-training quantization techniques.


