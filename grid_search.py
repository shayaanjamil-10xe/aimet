# %%
import os
import json
import numpy as np

# %% [markdown]
# ### Loading model

# %%
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from torchvision.models import resnet18
import torch
torch.manual_seed(0)

model = resnet18(pretrained=True)
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))
    
_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))

use_cuda

# %%
DATASET_DIR = '/home/shayan/Desktop/aimet/Examples/torch/quantization/'
import sys
sys.path.append("/home/shayan/Desktop/temp/aimet/")

import os
import torch
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

sys.path.remove("/home/shayan/Desktop/temp/aimet/")

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
    idx = 0
    with torch.no_grad():
        for path, input_data, target_data in data_loader:
            # if "cf135f199d8c7a9d0dce9aa35acfb4c70c14e0aa" not in path[0]:
            #     continue
            # if "cf" in path[0]:
            print(path)
            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)
            break


# %%
from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from copy import deepcopy

dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()

sim = QuantizationSimModel(model=deepcopy(model),
                           quant_scheme=QuantScheme.post_training_tf,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8,
                           config_file="/home/shayan/Desktop/aimet/my_config.json")

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)

# %%
# sim.export("./temp", filename_prefix="model", dummy_input=dummy_input)

# %%
print(str(sim))

# %%
offset = lambda min, delta: (min) / delta
delta = lambda max, min, bitwidth: (max - min) / (2 ** bitwidth - 1)

encoding = list(sim._get_qc_quantized_layers(sim.model)[0][1].param_quantizers.values())[0]._encoding[0]
encoding_min = encoding.min
encoding_max = encoding.max
encoding_delta = encoding.delta
encoding_offset = encoding.offset
encoding_bitwidth = list(sim._get_qc_quantized_layers(sim.model)[0][1].param_quantizers.values())[0].bitwidth

my_delta = delta(encoding_max, encoding_min, encoding_bitwidth)
my_offset = offset(encoding_min, my_delta)

# print(f"{my_delta} == {encoding_delta} and {my_offset} == {encoding_offset}")
# print(f"{encoding_min=}, {encoding_max=}, {encoding_delta=}")
# print(f"{encoding_offset=}, {encoding_bitwidth=}")

# %% [markdown]
# ### Time for Grid Search implementation

# %%
class UnlabelledDataset(torch.utils.data.Dataset):
    def __init__(self,):
        self.dataset = ImageNetDataPipeline.get_val_dataloader().dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][1]

# %%
offset = lambda min, delta: min / delta
delta = lambda max, min, bitwidth: (max - min) / (2 ** bitwidth - 1)

def compute_encoding(min, max, bw=8):
    enc_delta = delta(max, min, bw)
    enc_offset = offset(min, enc_delta)
    return {"min": min, "max": max, "offset": enc_offset, "delta": enc_delta}

def set_encoding_for_layer(layer, weight_encoding: dict, bias_encoding: dict=None, input_encoding: dict=None, output_encoding: dict=None):
    if weight_encoding:
        layer.param_quantizers['weight'].encoding.max = weight_encoding['max']
        layer.param_quantizers['weight'].encoding.min = weight_encoding['min']
        layer.param_quantizers['weight'].encoding.delta = weight_encoding['delta']
        layer.param_quantizers['weight'].encoding.offset = weight_encoding['offset']

    if bias_encoding:
        layer.param_quantizers['bias'].encoding.max = bias_encoding['max']
        layer.param_quantizers['bias'].encoding.min = bias_encoding['min']
        layer.param_quantizers['bias'].encoding.delta = bias_encoding['delta']
        layer.param_quantizers['bias'].encoding.offset = bias_encoding['offset']
            
    if input_encoding:
        for input_quantizer in layer.input_quantizers:
            input_quantizer.encoding.max = input_encoding['max']
            input_quantizer.encoding.min = input_encoding['min']
            input_quantizer.encoding.delta = input_encoding['delta']
            input_quantizer.encoding.offset = input_encoding['offset']
    
    if output_encoding:
        for output_quantizer in layer.output_quantizers:
            output_quantizer.encoding.max = output_encoding['max']
            output_quantizer.encoding.min = output_encoding['min']
            output_quantizer.encoding.delta = output_encoding['delta']
            output_quantizer.encoding.offset = output_encoding['offset']
        
    return layer
    
compute_encoding(0.0, 10, 8)

# %%
# quant_wrapper.input_quantizers[0].encoding.max

# %%
import torch
from aimet_torch import utils
from typing import Optional
num_batches = 1

def _compute_mse_loss(fp32_model: torch.nn.Module, quant_model: torch.nn.Module, verbose=False,
                      fp32_module:Optional[torch.nn.Module] = None, quant_wrapper:Optional[torch.nn.Module]=None) -> float:
    total = 0
    loss = 0.0
    batch_index = 0
    unlabeled_dataset_iterable = torch.utils.data.DataLoader(UnlabelledDataset(), batch_size=1, shuffle=False)
    for model_inputs in unlabeled_dataset_iterable:
        assert isinstance(model_inputs, (torch.Tensor, tuple, list))
        with torch.no_grad():
            if quant_wrapper and fp32_module:
                quant_outputs = quant_wrapper(quant_model(model_inputs)[:, :, 0, 0])
                fp32_outputs = fp32_module(fp32_model(model_inputs)[:, :, 0, 0])
            else:
                quant_outputs = quant_model(model_inputs)
                fp32_outputs = fp32_model(model_inputs)
        loss += torch.nn.functional.mse_loss(fp32_outputs, quant_outputs).item()
        total += fp32_outputs.size(0)
        batch_index += 1
        if batch_index == num_batches:
            break
        break

    average_loss = loss/total
    return average_loss

# %%
# import torch
# from aimet_torch import utils
# num_batches = 1

# def _compute_mse_loss(module: torch.nn.Module, quant_wrapper: torch.nn.Module,
#                         fp32_model: torch.nn.Module, sim: QuantizationSimModel, verbose=False) -> float:
#     """
#     Compute MSE loss between fp32 and quantized output activations for each batch, add for
#     all the batches and return averaged mse loss.

#     :param module: module from the fp32_model.
#     :param quant_wrapper: Corresponding quant wrapper from the QuantSim model.
#     :param fp32_model: PyTorch model.
#     :param sim: Quantsim model.
#     :return: MSE loss between fp32 and quantized output activations.
#     """
#     # output activations collector.
#     orig_module_collector = utils.ModuleData(fp32_model, module)
#     quant_module_collector = utils.ModuleData(sim.model, quant_wrapper)
    
#     if verbose:
#         weight = quant_module_collector._module.param_quantizers['weight']
#         bias = quant_module_collector._module.param_quantizers['bias']
#         inp = quant_module_collector._module.input_quantizers
#         outs = quant_module_collector._module.output_quantizers
        
#         print("WEIGHT")    
#         print(weight)
#         print("BIAS")
#         print(bias)
#         print("INPUT")
#         for i in inp:
#             print(i)
#         print("OUTPUT")
#         for i in outs:
#             print(i)
#     total = 0
#     loss = 0.0
#     batch_index = 0
#     unlabeled_dataset_iterable = torch.utils.data.DataLoader(UnlabelledDataset(), batch_size=1, shuffle=False)
#     for model_inputs in unlabeled_dataset_iterable:
#         assert isinstance(model_inputs, (torch.Tensor, tuple, list))
#         with torch.no_grad():
#             _, quantized_out_acts = quant_module_collector.collect_inp_out_data(model_inputs,
#                                                                                 collect_input=False,
#                                                                                 collect_output=True)
#             _, fp32_out_acts = orig_module_collector.collect_inp_out_data(model_inputs,
#                                                                             collect_input=False,
#                                                                             collect_output=True)
#         loss += torch.nn.functional.mse_loss(fp32_out_acts, quantized_out_acts).item()
#         total += fp32_out_acts.size(0)
#         batch_index += 1
#         if batch_index == num_batches:
#             break
#         break

#     average_loss = loss/total
#     return average_loss

# %%
fp32_name, fp32_module = list(dict(model.named_modules()).items())[-1]
fp32_model = torch.nn.Sequential(*list(model.children())[:-2])

quant_name, quant_wrapper = list(dict(sim.model.named_modules()).items())[-2]
quant_model = torch.nn.Sequential(*list(sim.model.children())[:-1])

print(fp32_module)
print(quant_wrapper)
print("-----------------------------")
print(fp32_name)
print(quant_name)

# %%
# [k for k in quant_wrapper.__dict__.keys() if "quantizers" in k]
# print(quant_wrapper.output_quantizers[0])

# %%
_compute_mse_loss(model, sim.model, None, quant_wrapper)

# %% [markdown]
# ### Updating Weight, Biases, etc at same time

# %%
from tqdm import tqdm
KEYS = ["weight", "bias", "input", "output"]
N = 30
def make_ranges(max, n=N, max_diff=0.1):
    max_diff = max * 0.5
    return np.linspace(max-max_diff if max-max_diff >= 0 else 0, max + max_diff, n)

def check_encodings_present(quant_wrapper):
    params_present, activations_present = [], []
    if quant_wrapper.param_quantizers['weight']._encoding:
        params_present.append('weight')
    if quant_wrapper.param_quantizers['bias']._encoding:
        params_present.append('bias')
    if quant_wrapper.input_quantizers[0]._encoding:
        activations_present.append('input')
    if quant_wrapper.output_quantizers[0]._encoding:
        activations_present.append('output')
    return params_present, activations_present

def grid_search(quant_model, fp32_model, fp32_module=None, quant_wrapper=None, verbose=False):
    params_present, activations_present = check_encodings_present(quant_wrapper) # list stating if encoding is present for weight, bias, input, output
    encoding_max = {} # dict containing the max value of the encoding for weight, bias, input, output
    for key in params_present:
        encoding_max[key] = getattr(quant_wrapper.param_quantizers[key]._encoding[0], 'max', None)
    for key in activations_present:
        encoding_max[key] = quant_wrapper.__dict__[f"{key}_quantizers"][0]._encoding[0].max
    from pprint import pprint
    pprint(encoding_max)
    encoding_ranges = {k: make_ranges(v) for k, v in encoding_max.items()}
    pprint(encoding_ranges)
    
    orig_loss = _compute_mse_loss(fp32_model, quant_model, fp32_module, quant_wrapper)
    
    if verbose:
        print(f"""
            weight_maximum = {encoding_max.get("weight", None)} \n {encoding_ranges.get("weight", None)} \n \n
            bias_maximum = {encoding_max.get("bias", None)} \n {encoding_ranges.get("bias", None)} \n \n
            input_maximum = {encoding_max.get("input", None)} \n {encoding_ranges.get("input", None)} \n \n
            output_maximum = {encoding_max.get("output", None)} \n {encoding_ranges.get("output", None)} \n \n
            """)
        print(f"Original loss: {orig_loss}")

    best_loss = orig_loss
    best_params = deepcopy(encoding_max)
    
    all_logs = []
    for weight in tqdm(encoding_ranges.get("weight", [])):
        for bias in tqdm(encoding_ranges.get("bias", [])):
            for inp in encoding_ranges.get("output", []):
                weight_encoding = compute_encoding(-weight, weight)
                bias_encoding = compute_encoding(-bias, bias)
                input_encoding = compute_encoding(-inp, inp)
                set_encoding_for_layer(quant_wrapper, weight_encoding, bias_encoding, output_encoding=input_encoding)
                loss = _compute_mse_loss(fp32_model, quant_model, fp32_module, quant_wrapper)
                all_logs.append({"weight": weight, "bias": bias, "output": inp, "loss": loss})
                if loss < best_loss:
                    print(f"New best loss: {loss} with params: {weight}, {bias}, {inp}")
                    best_loss = loss
                    best_params = {"weight": weight, "bias": bias, "output": inp}
    
    if verbose:
        if best_loss != orig_loss:
            print(f"New best loss: {best_loss} with params: {best_params}")
        else:
            print(f"No better params found. Original loss: {orig_loss}")
        if encoding_max.get("weight", None) != best_params.get('weight', None):
            print(f"Weight maximum changed from {encoding_max.get('weight', None)} to {best_params.get('weight', None)}")
        if encoding_max.get("bias", None) != best_params.get('bias', None):
            print(f"Bias maximum changed from {encoding_max.get('bias', None)} to {best_params.get('bias', None)}")
        if encoding_max.get("input", None) != best_params.get('input', None):
            print(f"input maximum changed from {encoding_max.get('input', None)} to {best_params.get('input', None)}")
        if encoding_max.get("output", None) != best_params.get('output', None):
            print(f"output maximum changed from {encoding_max.get('output', None)} to {best_params.get('output', None)}")
    
    return best_loss, best_params, all_logs
    
best_loss, best_params, all_logs =  grid_search(sim.model, model, None, quant_wrapper, verbose=True)

# %%
fp32_name += "_new_mse_loss_big_range"
fp32_name

# %%
import pandas as pd
import json
import os

output_dir = "./logs/"
output_dir = os.path.join(output_dir, fp32_name)

os.makedirs(output_dir, exist_ok=True)
pd.DataFrame(all_logs).to_csv(os.path.join(output_dir, "logs.csv"))

with open(os.path.join(output_dir, "best_params.json"), "w") as f:
    best_params.update({"best_loss": best_loss})
    json.dump(best_params, f)


# %%
# from tqdm import tqdm
# def make_ranges(max, n=15, max_diff=0.1):
#     return np.linspace(max-max_diff if max-max_diff >= 0 else 0, max, n)

# def grid_search(sim, model, fp32_module, quant_wrapper, verbose=False):
#     weight_maximum = quant_wrapper.param_quantizers['weight']._encoding[0].max
#     bias_maximum = quant_wrapper.param_quantizers['bias']._encoding[0].max
#     input_maximum = quant_wrapper.input_quantizers[0]._encoding[0].max
#     output_maximum = [out._encoding for out in quant_wrapper.output_quantizers if out._encoding]
    
#     weight_range = make_ranges(weight_maximum)
#     bias_range = make_ranges(bias_maximum)
#     input_range = make_ranges(input_maximum)
    
#     orig_loss = _compute_mse_loss(fp32_module, quant_wrapper, model, sim)
    
#     if verbose:
#         print(f"""
#             weight_maximum = {weight_maximum} \n {weight_range} \n \n
#             bias_maximum = {bias_maximum} \n {bias_range} \n \n
#             input_maximum = {input_maximum} \n {input_range} \n \n
#             output_maximum = {output_maximum} \n None \n \n
#             """)
#         print(f"Original loss: {orig_loss}")
        
#     best_loss = float('inf')
#     best_params = {"weight": weight_maximum, "bias": bias_maximum, "input": input_maximum}
#     for weight in tqdm(weight_range):
#         for bias in tqdm(bias_range):
#             for inp in input_range:
#                 weight_encoding = compute_encoding(-weight, weight)
#                 bias_encoding = compute_encoding(-bias, bias)
#                 input_encoding = compute_encoding(-inp, inp)
#                 set_encoding_for_layer(quant_wrapper, weight_encoding, bias_encoding, input_encoding)
#                 loss = _compute_mse_loss(fp32_module, quant_wrapper, model, sim)
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_params = {"weight": weight, "bias": bias, "input": inp}
    
#     if verbose:
#         print(f"New best loss: {best_loss} with params: {best_params}")
#         if weight_maximum != best_params['weight']:
#             print(f"Weight maximum changed from {weight_maximum} to {best_params['weight']}")
#         if bias_maximum != best_params['bias']:
#             print(f"Bias maximum changed from {bias_maximum} to {best_params['bias']}")
#         if input_maximum != best_params['input']:
#             print(f"Input maximum changed from {input_maximum} to {best_params['input']}")
        
    
# grid_search(sim, model, fp32_module, quant_wrapper, verbose=True)

# %%
from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from copy import deepcopy

dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()

tf_copy = deepcopy(model)
tf_enhanced_sim = QuantizationSimModel(model=tf_copy,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8,
                           config_file="/home/shayan/Desktop/aimet/my_config.json")

tf_enhanced_sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)

# %%
fp32_name, fp32_module = list(dict(tf_copy.named_modules()).items())[-1]
quant_name, quant_wrapper = list(dict(tf_enhanced_sim.model.named_modules()).items())[-2]

print("TF Enhanced QUANTIZATION INFO")
print(quant_wrapper)
print(fp32_module)
print("-----------------------------")
print(f"{quant_name=} {fp32_name=}")
print("-----------------------------")
print("WEIGHT")
print(quant_wrapper.param_quantizers['weight'])
print("BIAS")
print(quant_wrapper.param_quantizers['bias'])
print("INPUT")
for i in quant_wrapper.input_quantizers:
    print(i)
print("OUTPUT")
for i in quant_wrapper.output_quantizers:
    print(i)

# %%
tf_enhanced_loss = _compute_mse_loss(tf_copy, tf_enhanced_sim.model, None, quant_wrapper)
print(f"TF Enhanced Loss: {tf_enhanced_loss}")


