# %%
DATASET_DIR = '/home/shayan/Desktop/aimet/Examples/torch/quantization/'         # Replace this path with a real directory

# %% [markdown]
# ---
# ## 1. Instantiate the example training and validation pipeline
# 
# **Use the following training and validation loop for the image classification task.**
# 
# Things to note:
# 
# - AIMET does not put limitations on how the training and validation pipeline is written. AIMET modifies the user's model to create a QuantizationSim model, which is still a PyTorch model. The QuantizationSim model can be used in place of the original model when doing inference or training.
# - AIMET doesn not put limitations on the interface of the `evaluate()` or `train()` methods. You should be able to use your existing evaluate and train routines as-is.
# 

# %%
# ! pip install git+https://github.com/modestyachts/ImageNetV2_pytorch


# from imagenetv2_pytorch import ImageNetV2Dataset


# images = ImageNetV2Dataset()

# %%
import sys
sys.path.append("/home/shayan/Desktop/aimet/")

import os
import torch
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

sys.path.remove("/home/shayan/Desktop/aimet/")

class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
                                         image_size=image_net_config.dataset['image_size'],
                                         batch_size=image_net_config.evaluation['batch_size'],
                                         is_training=False,
                                         num_workers=image_net_config.evaluation['num_workers']).data_loader
        return data_loader

    @staticmethod
    def evaluate(model: torch.nn.Module, use_cuda: bool) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param model: the model to evaluate
        :param iterations: the number of batches to be used to evaluate the model. A value of 'None' means the model will be
                           evaluated on the entire dataset once.
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations=None, use_cuda=use_cuda)

# %% [markdown]
# ---
# 
# ## 2. Load the model and evaluate to get a baseline FP32 accuracy score

# %% [markdown]
# **2.1 Load a pretrained resnet18 model from torchvision.** 
# 
# You can load any pretrained PyTorch model instead.

# %%
from torchvision.models import resnet18

model = resnet18(pretrained=True)

# %% [markdown]
# AIMET quantization simulation requires the model definition to follow certain guidelines. For example, functionals defined in the forward pass should be changed to the equivalent **torch.nn.Module**.
# The [AIMET user guide](https://quic.github.io/aimet-pages/releases/latest/user_guide/index.html) lists all these guidelines.
# 
# **2.2 Use the following ModelPreparer API call to automate the model definition changes required to comply with the AIMET guidelines.** 
# 
# The call uses the graph transformation feature available in PyTorch 1.9+.

# %%
from aimet_torch.model_preparer import prepare_model

model = prepare_model(model)

# %% [markdown]
# ---
# 
# **2.3 Decide whether to place the model on a CPU or CUDA device.** 
# 
# This example uses CUDA if it is available. You can change this logic and force a device placement if needed.

# %%
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))
use_cuda

# %% [markdown]
# ---
# 
# **2.4 Compute the floating point 32-bit (FP32) accuracy of this model using the evaluate() routine.**

# %%
# accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)
# print(accuracy)

# %% [markdown]
# ---
# 
# ## 3. Create a quantization simulation model and determine quantized accuracy
# 
# ### Fold Batch Norm layers
# 
# Before calculating the simulated quantized accuracy using QuantizationSimModel, fold the BatchNorm (BN) layers into adjacent Convolutional layers. The BN layers that cannot be folded are left as they are.
# 
# BN folding improves inference performance on quantized runtimes but can degrade accuracy on these platforms. This step simulates this on-target drop in accuracy. 
# 
# **3.1 Use the following code to call AIMET to fold the BN layers in-place on the given model.**

# %%
from aimet_torch.batch_norm_fold import fold_all_batch_norms

_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))

# %%
len(_)

# %% [markdown]
# ### Create the Quantization Sim Model
# 
# **3.2 Use AIMET to create a QuantizationSimModel.**
# 
#  In this step, AIMET inserts fake quantization ops in the model graph and configures them.
# 
# Key parameters:
# 
# - Setting **default_output_bw** to 8 performs all activation quantizations in the model using integer 8-bit precision
# - Setting **default_param_bw** to 8 performs all parameter quantizations in the model using integer 8-bit precision
# - **num_batches** is the number of batches to use to compute encodings. Only five batches are used here for the sake of speed
# 
# See [QuantizationSimModel in the AIMET API documentation](https://quic.github.io/aimet-pages/AimetDocs/api_docs/torch_quantsim.html#aimet_torch.quantsim.QuantizationSimModel.compute_encodings) for a full explanation of the parameters.

# %%
from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel

dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()

sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

# %% [markdown]
# ---
# **3.3 Print the model to verify the modifications AIMET has made. **
# 
# Note that AIMET has added quantization wrapper layers. 
# 
# <div class="alert alert-info">
# 
# Note
# 
# Use sim.model to access the modified PyTorch model. By default, AIMET creates a copy of the original model prior to modifying it. There is a parameter to override this behavior.
# 
# </div>

# %%
print(sim.model)

# %% [markdown]
# ---
# Note also that AIMET has configured the added fake quantization nodes, which AIMET refers to as "quantizers". 
# 
# **3.4 Print the sim object to see the quantizers.**

# %%
print(sim)

# %% [markdown]
# ---
# AIMET has added quantizer nodes to the model graph, but before the sim model can be used for inference or training, scale and offset quantization parameters must be calculated for each quantizer node by passing unlabeled data samples through the model to collect range statistics. This process is sometimes referred to as calibration. AIMET refers to it as "computing encodings".
# 
# **3.5 Create a routine to pass unlabeled data samples through the model.** 
# 
# The following code is one way to write a routine that passes unlabeled samples through the model to compute encodings. It uses the existing train or validation data loader to extract samples and pass them to the model. Since there is no need to compute loss metrics, it ignores the model output.  


# %%
def pass_calibration_data(sim_model, use_cuda):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for path, input_data, target_data in data_loader:
            print(path)
            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)
            break
            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break

# %% [markdown]
# A few notes regarding the data samples:
# 
# - A very small percentage of the data samples are needed. For example, the training dataset for ImageNet has 1M samples; 500 or 1000 suffice to compute encodings.
# - The samples should be reasonably well distributed. While it's not necessary to cover all classes, avoid extreme scenarios like using only dark or only light samples. That is, using only pictures captured at night, say, could skew the results.
# 
# ---
# 
# **3.6 Call AIMET to use the routine to pass data through the model and compute the quantization encodings.** 
# 
# Encodings here refer to scale and offset quantization parameters.

# %%
sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)
