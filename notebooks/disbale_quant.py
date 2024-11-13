#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import json
from mmcv.transforms import Compose
import numpy as np
from mmdet.utils import get_test_pipeline_cfg

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def preprocess(test_pipeline, image):
    if isinstance(image, np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)
    return test_pipeline(dict(img=image))

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_json_path, transform=None):
        self.transform = transform
        self.images_dir = images_dir
        self.annotations_json = read_json(annotations_json_path)


    def __len__(self):
        return len(self.annotations_json['images'])

    def __getitem__(self, idx):
        image_dict = self.annotations_json['images'][idx]
        image_path = os.path.join(self.images_dir, image_dict['file_name'])
        image_id = image_dict['id']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed_images = self.transform(image)
        else:
            transformed_images = image

        return image_id, image_path, transformed_images


# calibrationDataloader = DataLoader(calibrationDataset, batch_size=32, shuffle=True)


# In[2]:


import torch
from mmdet.apis import DetInferencer

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([640, 640]),  # Resize
])

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = '/teamspace/studios/this_studio/mmdetection/rtmdet_tiny_8xb32-300e_coco.py'
WEIGHTS_PATH = '/teamspace/studios/this_studio/mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
EVAL_DATASET_SIZE = 5000
CALIBRATION_DATASET_SIZE = 1000
BATCH_SIZE = 80

ROOT_DATASET_DIR = '/teamspace/studios/this_studio/COCO'
IMAGES_DIR = os.path.join(ROOT_DATASET_DIR, 'images')
ANNOTATIONS_JSON_PATH = os.path.join(ROOT_DATASET_DIR, 'annotations/instances_val2017.json')
# ANNOTATIONS_JSON_PATH = "/home/shayaan/Desktop/aimet/my_mmdet/temp.json"

model = DetInferencer(model=CONFIG_PATH, weights=WEIGHTS_PATH, device=DEVICE)
evalDataset = CustomImageDataset(images_dir=IMAGES_DIR, annotations_json_path=ANNOTATIONS_JSON_PATH, transform=transform)
eval_data_loader = DataLoader(evalDataset, batch_size=BATCH_SIZE)


# In[3]:


total_params = sum(p.numel() for p in model.model.parameters())
total_params / 10 ** 6, len(list(model.model.modules())) - 1


# In[4]:


from mmcv.transforms import Compose
test_evaluator = model.cfg.test_evaluator
test_evaluator.type = 'mmdet.evaluation.CocoMetric' 
test_evaluator.dataset_meta = model.model.dataset_meta
test_evaluator.ann_file = ANNOTATIONS_JSON_PATH
test_evaluator = Compose(test_evaluator)


# In[5]:


import random
from typing import Optional
from tqdm import tqdm
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset

from mmengine.structures import InstanceData
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.registry import MODELS

collate_preprocessor = model.preprocess
predict_by_feat = model.model.bbox_head.predict_by_feat
rescale = True

preprocessor = MODELS.build(model.cfg.model.data_preprocessor)
def add_pred_to_datasample(data_samples, results_list):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


# In[6]:


def eval_callback(model, use_cuda):
    data_loader = eval_data_loader
    new_preds = []
    for image_id, image_path, _ in tqdm(data_loader):
        pre_processed = collate_preprocessor(inputs=image_path, batch_size=BATCH_SIZE)
        _, data = list(pre_processed)[0]
        data = preprocessor(data, False)
        preds = model(data['inputs'].cuda())
        batch_img_metas = [
        data_samples.metainfo for data_samples in data['data_samples']
        ]
        preds = predict_by_feat(*preds, batch_img_metas=batch_img_metas, rescale=True)
        preds = add_pred_to_datasample(data['data_samples'], preds)
        
        for img_id, pred in zip(image_id, preds):
            pred = pred.pred_instances
            new_pred = InstanceData(metainfo={"img_id": int(img_id)})
            new_pred.bboxes = [np.array(p) for p in pred['bboxes'].cpu()]
            new_pred.labels = pred['labels'].cpu()
            new_pred.scores = pred['scores'].cpu()
            new_preds.append(new_pred)

    eval_results = test_evaluator(new_preds)
    # num_file = len(glob("/home/shayaan/Desktop/aimet/aimet/Examples/torch/quantization/quant_anal_eval_stats/eval_acc_quant_*"))
    # with open(f"/home/shayaan/Desktop/aimet/aimet/Examples/torch/quantization/quant_anal_eval_stats/eval_acc_quant_{num_file}.json", "w") as f:
    #     json.dump(eval_results, f, indent=4)
    bbox_map = eval_results['bbox_mAP']
    return bbox_map


# In[7]:


def pass_calibration_data(model: torch.nn.Module, use_cuda):
    data_loader = eval_data_loader
    batch_size = data_loader.batch_size
    model.eval()
    samples = CALIBRATION_DATASET_SIZE
    batch_ctr = 0
    with torch.no_grad():
        for image_id, image_path, _ in tqdm(data_loader):
            pre_processed = collate_preprocessor(inputs=image_path, batch_size=BATCH_SIZE)
            _, data = list(pre_processed)[0]
            data = preprocessor(data, False)
            
            preds = model(data['inputs'].cuda())

            batch_ctr += 1
            if (batch_ctr * batch_size) > samples:
                break  


# AIMET quantization simulation requires the user's model definition to follow certain guidelines.
# For example, functionals defined in forward pass should be changed to equivalent torch.nn.Module.
# AIMET user guide lists all these guidelines.
# 
# The following **ModelPreparer** API uses new graph transformation feature available in PyTorch 1.9+ version and automates model definition changes required to comply with the above guidelines.

# In[8]:


from aimet_torch.model_preparer import prepare_model

model = prepare_model(model.model)


# ---
# We should decide whether to place the model on a CPU or CUDA device.
# This example code will use CUDA if available in your current execution environment.
# You can change this logic and force a device placement if needed.

# In[ ]:


# print("------------FP 32 MODEL MODULES ------------")
# print(dict(model.named_modules()))


# In[9]:


use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))
use_cuda


# ---
# 
# ## 3. Apply QuantAnalyzer to the model
# 
# QuantAnalyzer requires two functions to be defined by the user for passing data through the model:
# 
# **Forward pass callback**
# 
# One function will be used to pass representative data through a quantized version of the model to calibrate quantization parameters.
# This function should be fairly simple - use the existing train or validation data loader to extract some samples and pass them to the model.
# We don't need to compute any loss metrics, so we can just ignore the model output.
# 
# The function **must** take two arguments, the first of which will be the model to run the forward pass on.
# The second argument can be anything additional which the function requires to run, and can be in the form of a single item or a tuple of items.
# 
# If no additional argument is needed, the user can specify a dummy "_" parameter for the function.
# 
# A few pointers regarding the forward pass data samples:
# 
# - In practice, we need a very small percentage of the overall data samples for computing encodings.
#   For example, the training dataset for ImageNet has 1M samples. For computing encodings we only need 500 to 1000 samples.
# - It may be beneficial if the samples used for computing encoding are well distributed.
#   It's not necessary that all classes need to be covered since we are only looking at the range of values at every layer activation.
#   However, we definitely want to avoid an extreme scenario like all 'dark' or 'light' samples are used - e.g. only using pictures captured at night might not give ideal results.
# 
# The following shows an example of a routine that passes unlabeled samples through the model for computing encodings.
# This routine can be written in many ways; this is just an example.
# This function only requires unlabeled data as no loss or other evaluation metric is needed.

# In order to pass this function to QuantAnalyzer, we need to wrap it in a CallbackFunc object, as shown below.
# The CallbackFunc takes two arguments: the callback function itself, and the inputs to pass into the callback function.

# In[10]:


from aimet_torch.quant_analyzer import CallbackFunc

forward_pass_callback = CallbackFunc(pass_calibration_data, use_cuda)


# ---
# 
# **Evaluation callback**
# 
# The second function will be used to evaluate the model, and needs to return an accuracy metric.
# In here, the user should pass any amount of data through the model which they would like when evaluating their model for accuracy.
# 
# Like the forward pass callback, this function also must take exactly two arguments: the model to evaluate, and any additional argument needed for the function to work.
# The second argument can be a tuple of items in case multiple items are needed.
# 
# We will be using the ImageNetDataPipeline's evaluate defined above for this purpose.
# Like the forward pass callback, we need to wrap the evaluation callback in a CallbackFunc object as well.

# In[11]:


eval_callback = CallbackFunc(eval_callback, use_cuda)


# ---
# 
# **Enabling MSE loss per layer analysis**
# 
# An optional analysis step in QuantAnalyzer calculates the MSE loss per layer in the model, comparing the layer outputs from the original FP32 model vs. a quantized model.
# To perform this step, the user needs to also provide an unlabeled DataLoader to QuantAnalyzer.
# 
# We will demonstrate this step by using the ImageNetDataLoader imported above.

# In[12]:


data_loader = eval_data_loader


# ---
# 
# QuantAnalyzer also requires a dummy input to the model.
# This dummy input does not need to be representative of the dataset.
# All that matters is that the input shape is correct for the model to run on.

# In[13]:


dummy_input = torch.rand(1, 3, 640, 640).cuda()    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()


# ---
# We are now ready to apply QuantAnalyzer.

# In[ ]:


module_names = dict(model.named_modules())
modules_to_ignore = ['backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_14', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_7', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.conv', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_21', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_30', 'neck.top_down_blocks.0.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_37', 'neck.top_down_blocks.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_44', 'neck.bottom_up_blocks.0.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_51', 'neck.bottom_up_blocks.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_58']
modules_to_ignore = [module_names[m] in list(module_names.keys()) for m in modules_to_ignore]


# In[14]:


from aimet_torch.v2.quant_analyzer import QuantAnalyzer

quant_analyzer = QuantAnalyzer(model, dummy_input, forward_pass_callback, eval_callback, modules_to_ignore)


# In[ ]:


# bn_module = dict(model.named_modules())['backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_14']
# print(type(bn_module))


# In[15]:


from aimet_common.defs import QuantScheme

sim = quant_analyzer._create_quantsim_and_encodings(quant_scheme=QuantScheme.post_training_tf_enhanced,
                       default_param_bw=8,
                       default_output_bw=8,
                       config_file=None)


# In[ ]:





# In[ ]:


# print("------------Quant Sim MODEL MODULES ------------")
# print(dict(sim.model.named_modules()))


# To enable the MSE loss analysis, we set the following:

# In[14]:


# quant_analyzer.enable_per_layer_mse_loss(data_loader, num_batches=4)


# Finally, to start the analyzer, we call .analyze().
# 
# A few of the parameters are explained here:
# - **quant_scheme**:
#     - We set this to "post_training_tf_enhanced"
#       With this choice of quant scheme, AIMET will use the TF Enhanced quant scheme to initialize the quantization parameters like scale/offset.
# - **default_output_bw**: Setting this to 8 means that we are asking AIMET to perform all activation quantizations in the model using integer 8-bit precision.
# - **default_param_bw**: Setting this to 8 means that we are asking AIMET to perform all parameter quantizations in the model using integer 8-bit precision.
# 
# There are other parameters that are set to default values in this example.
# Please check the AIMET API documentation of QuantizationSimModel to see reference documentation for all the parameters.
# 
# When you call the analyze method, the following analyses are run:
# 
# - Compare fp32 accuracy, accuracy with only parameters quantized, and accuracy with only activations quantized
# - For each layer, track the model accuracy when quantization for all other layers is disabled (enabling quantization for only one layer in the model at a time)
# - For each layer, track the model accuracy when quantization for all other layers is enabled (disabling quantization for only one layer in the model at a time)
# - Track the minimum and maximum encoding parameters calculated by each quantizer in the model as a result of forward passes through the model with representative data
# - When the TF Enhanced quantization scheme is used, track the histogram of tensor ranges seen by each quantizer in the model as a result of forward passes through the model with representative data
# - If enabled, track the MSE loss seen at each layer by comparing layer outputs of the original fp32 model vs. a quantized model

# In[15]:


# from aimet_common.defs import QuantScheme

# quant_analyzer.analyze(quant_scheme=QuantScheme.post_training_tf_enhanced,
#                        default_param_bw=8,
#                        default_output_bw=8,
#                        config_file=None,
#                        results_dir="./tmp/")


# AIMET will also output .html plots and json files where appropriate for each analysis to help visualize the data.
# 
# The following output files will be produced, in a folder specified by the user:
# Output directory structure will be like below
# 
# ```
# results_dir
# |-- per_layer_quant_enabled.html
# |-- per_layer_quant_enabled.json
# |-- per_layer_quant_disabled.html
# |-- per_layer_quant_disabled.json
# |-- min_max_ranges
# |   |-- activations.html
# |   |-- activations.json
# |   |-- weights.html
# |   +-- weights.json
# |-- activations_pdf
# |   |-- name_{input/output}_{index_0}.html
# |   |-- name_{input/output}_{index_1}.html
# |   |-- ...
# |   +-- name_{input/output}_{index_N}.html
# |-- weights_pdf
# |   |-- layer1
# |   |   |-- param_name_{channel_index_0}.html
# |   |   |-- param_name_{channel_index_1}.html
# |   |   |-- ...
# |   |   +-- param_name_{channel_index_N}.html
# |   |-- layer2
# |   |   |-- param_name_{channel_index_0}.html
# |   |   |-- param_name_{channel_index_1}.html
# |   |   |-- ...
# |   |   +-- param_name_{channel_index_N}.html
# |   |-- ...
# |   |-- layerN
# |   |   |-- param_name_{channel_index_0}.html
# |   |   |-- param_name_{channel_index_1}.html
# |   |   |-- ...
# |   +-- +-- param_name_{channel_index_N}.html
# |-- per_layer_mse_loss.html
# +-- per_layer_mse_loss.json
# ```
# 
# #### Per-layer analysis by enabling/disabling quantization wrappers
# 
# - per_layer_quant_enabled.html: A plot with layers on the x-axis and model accuracy on the y-axis, where each layer's accuracy represents the model accuracy when all quantizers in the model are disabled except for that layer's parameter and activation quantizers.
# - per_layer_quant_enabled.json: A json file containing the data shown in per_layer_quant_enabled.html, associating layer names with model accuracy.
# - per_layer_quant_disabled.html: A plot with layers on the x-axis and model accuracy on the y-axis, where each layer's accuracy represents the model accuracy when all quantizers in the model are enabled except for that layer's parameter and activation quantizers.
# - per_layer_quant_disabled.json: A json file containing the data shown in per_layer_quant_disabled.html, associating layer names with model accuracy.
# 
# ![per_layer_quant_enabled.html](./images/quant_analyzer_per_layer_quant_enabled.PNG)
# 
# #### Encoding min/max ranges
# 
# - min_max_ranges: A folder containing the following sets of files:
#     - activations.html: A plot with output activations on the x-axis and min-max values on the y-axis, where each output activation's range represents the encoding min and max parameters computed during forward pass calibration (explained below).
#     - activations.json: A json file containing the data shown in activations.html, associating layer names with min and max encoding values.
#     - weights.html: A plot with parameter names on the x-axis and min-max values on the y-axis, where each parameter's range represents the encoding min and max parameters computed during forward pass calibration.
#     - weights.json: A json file containing the data shown in weights.html, associating parameter names with min and max encoding values.
# 
# ![min_max_ranges.html](./images/quant_analyzer_min_max_ranges.PNG)
# 
# #### PDF of statistics
# 
# - (If TF Enhanced quant scheme is used) activations_pdf: A folder containing html files for each layer, plotting the histogram of tensor values seen for that layer's output activation seen during forward pass calibration.
# - (If TF Enhanced quant scheme is used) weights_pdf: A folder containing sub folders for each layer with weights.
#   Each layer's folder contains html files for each parameter of that layer, with a histogram plot of tensor values seen for that parameter seen during forward pass calibration.
# 
# ![weights_pdf.html](./images/quant_analyzer_weights_pdf.PNG)
# 
# #### Per-layer MSE loss
# - (Optional, if per layer MSE loss is enabled) per_layer_mse_loss.html: A plot with layers on the x-axis and MSE loss on the y-axis, where each layer's MSE loss represents the MSE seen comparing that layer's outputs in the FP32 model vs. the quantized model.
# - (Optional, if per layer MSE loss is enabled) per_layer_mse_loss.json: A json file containing the data shown in per_layer_mse_loss.html, associating layer names with MSE loss.
# 
# ![per_layer_mse_loss.html](./images/quant_analyzer_per_layer_mse_loss.PNG)
