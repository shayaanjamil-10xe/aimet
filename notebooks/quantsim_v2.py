#!/usr/bin/env python
# coding: utf-8

# In[63]:


import cv2
import os
import torch
import json
import numpy as np
from tqdm.notebook import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def read_txt(txt_path):
    with open(txt_path) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
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


# In[116]:


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
BATCH_SIZE = 64

ROOT_DATASET_DIR = '/teamspace/studios/this_studio/COCO'
IMAGES_DIR = os.path.join(ROOT_DATASET_DIR, 'images')
ANNOTATIONS_JSON_PATH = os.path.join(ROOT_DATASET_DIR, 'annotations/instances_val2017.json')
# ANNOTATIONS_JSON_PATH = "/home/shayaan/Desktop/aimet/my_mmdet/temp.json"

model = DetInferencer(model=CONFIG_PATH, weights=WEIGHTS_PATH, device=DEVICE)
evalDataset = CustomImageDataset(images_dir=IMAGES_DIR, annotations_json_path=ANNOTATIONS_JSON_PATH, transform=transform)
eval_data_loader = DataLoader(evalDataset, batch_size=BATCH_SIZE)
calibration_images = read_txt('/teamspace/studios/this_studio/aimet/Examples/torch/quantization/calibration_image_ids.txt')
calibration_data_loader = DataLoader(calibration_images, batch_size=BATCH_SIZE)
DEVICE


# In[27]:


# import inspect

# lines = inspect.getsource(dict(model.model.named_modules())['backbone.stem.1.bn'].__class__)
# print(lines)


# In[65]:


from collections import OrderedDict
from copy import deepcopy

def replace_rtm_bn(model):
    m = deepcopy(model.model)

    def is_leaf(module): 
        return len(module._modules) == 0

    def replace_bn(m):

        if is_leaf(m):
            return 

        for _, child in m.named_children(): 
            
            if "bn" in child._modules.keys():
                bn = child._modules.get("bn")
                bn_params = deepcopy(bn._parameters)
                bn_buffers = deepcopy(bn._buffers)
                new_bn = torch.nn.BatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=bn.affine, track_running_stats=bn.track_running_stats)
                new_bn._parameters["weight"].data = bn_params["weight"].data
                new_bn._parameters["bias"].data = bn_params["bias"].data
                new_bn._buffers["running_mean"].data = bn_buffers["running_mean"].data
                new_bn._buffers["running_var"].data = bn_buffers["running_var"].data
                new_bn._buffers["num_batches_tracked"].data = bn_buffers["num_batches_tracked"].data
                child._modules["bn"] = new_bn
                
            replace_bn(child)

    replace_bn(m)

    return m


# In[29]:


from tqdm import tqdm
import torch

from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.registry import MODELS
from mmcv.transforms import Compose

test_evaluator = model.cfg.test_evaluator
test_evaluator.type = 'mmdet.evaluation.CocoMetric' 
test_evaluator.dataset_meta = model.model.dataset_meta
test_evaluator.ann_file = ANNOTATIONS_JSON_PATH
test_evaluator = Compose(test_evaluator)

collate_preprocessor = model.preprocess
predict_by_feat = model.model.bbox_head.predict_by_feat
rescale = True

preprocessor = MODELS.build(model.cfg.model.data_preprocessor)
def add_pred_to_datasample(data_samples, results_list):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


# In[30]:


def pass_calibration_data(model: torch.nn.Module, samples: int):
    data_loader = eval_data_loader
    batch_size = data_loader.batch_size
    model.eval()
    batch_ctr = 0
    with torch.no_grad():
        for image_path in tqdm(calibration_data_loader):
            image_path = [os.path.join(IMAGES_DIR, x) for x in image_path]
            pre_processed = collate_preprocessor(inputs=image_path, batch_size=batch_size)
            _, data = list(pre_processed)[0]
            data = preprocessor(data, False)
            
            preds = model(data['inputs'].to(DEVICE))  

            # batch_ctr += 1
            # if (batch_ctr * batch_size) > samples:
            #     break


# In[15]:


dict(model.model.named_modules())['backbone.stage1.1.main_conv.bn'].weight


# In[17]:


# from aimet_torch.model_preparer import prepare_model

# class CustomBatchNorm2d(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.bn = torch.nn.BatchNorm2d(num_features = 12, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
    
#     def forward(self, x):
#         return self.bn(x)
    
# class CustomBatchNorm2d(torch.nn.modules.batchnorm._BatchNorm):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _check_input_dim(self, input: torch.Tensor):
#         return

# class CustomModel(torch.nn.Module):
#     def __init__(self): 
#         super().__init__()

#         self.conv = torch.nn.Conv2d(3, 12, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.bn = CustomBatchNorm2d(num_features = 12, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x
    

# custom_model = CustomModel()

# model = prepare_model(custom_model)

# from aimet_torch.batch_norm_fold import fold_all_batch_norms

# fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))


# In[66]:





# In[32]:


# len(bn_pairs)


# In[16]:


from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms

def exclude_modules_from_quant(sim, modules_to_ignore):
    name_to_quant_wrapper_dict = {}
    for name, module in sim.model.named_modules():
        name_to_quant_wrapper_dict[name] = module

    quant_wrappers_to_ignore = []
    for name in modules_to_ignore:
        quant_wrapper = name_to_quant_wrapper_dict[name]
        quant_wrappers_to_ignore.append(quant_wrapper)

    sim.exclude_layers_from_quantization(quant_wrappers_to_ignore)

dummy_input = torch.rand(1, 3, 640, 640).to(DEVICE)# Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)

m = replace_rtm_bn(model)
m = prepare_model(m)
bn_pairs = fold_all_batch_norms(m, input_shapes=(1, 3, 640, 640))
print("Length of bn pairs: ", len(bn_pairs))

modules = dict(m.named_modules())

modules_to_change = ['backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.conv', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn', 'neck.top_down_blocks.0.blocks.0.conv2.depthwise_conv.bn', 'neck.top_down_blocks.1.blocks.0.conv2.depthwise_conv.bn', 'neck.bottom_up_blocks.0.blocks.0.conv2.depthwise_conv.bn', 'neck.bottom_up_blocks.1.blocks.0.conv2.depthwise_conv.bn']
modules_to_change = {modules[x]: {"output_bw": 16, "param_bw": 16, "data_type": QuantizationDataType.float, "module_name": x} for x in modules_to_change}

print("dtype of model: ", list(dict(m.named_parameters()).values())[0].dtype)
quant_sim = QuantizationSimModel(model=m,
                                quant_scheme=QuantScheme.post_training_tf_enhanced,
                                default_param_bw=8,
                                default_output_bw=8,
                                config_file=None,
                                dummy_input=dummy_input,
                                modules_to_change=modules_to_change,
                                in_place=True)

### if load encodings
# quant_sim.load_encodings(encodings="/teamspace/studios/this_studio/aimet/Examples/torch/quantization/sim_model_excluded_modules/rtm_det_torch.encodings")
# quant_sim.load_encodings(encodings="/teamspace/studios/this_studio/aimet/Examples/torch/quantization/quant_scheme_W@tf / A@tf/rtm_det_torch.encodings")
# quant_sim.load_encodings(encodings=f"{BASE_PATH}/rtm_det_torch.encodings")

### else compute encodings
# quant_sim.compute_encodings(pass_calibration_data, 1000)

# modules_to_ignore = ['backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_14', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_7', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.conv', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_21', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_30', 'neck.top_down_blocks.0.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_37', 'neck.top_down_blocks.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_44', 'neck.bottom_up_blocks.0.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_51', 'neck.bottom_up_blocks.1.blocks.0.conv2.depthwise_conv.bn.module_batch_norm_58']
# exclude_modules_from_quant(quant_sim, modules_to_ignore)


# In[18]:


print(str(quant_sim))


# In[19]:


### else compute encodings
quant_sim.compute_encodings(pass_calibration_data, 1000)


# In[20]:


import torch
import os
import traceback
import shutil
dummy_input = torch.rand(1, 3, 640, 640)
output_dir = f"/teamspace/studios/this_studio/aimet/exported_models/bn_folded_fp16_encodings"
os.makedirs(output_dir, exist_ok=True)
quant_sim.export(path=output_dir,
            filename_prefix="rtm_det",
            dummy_input=dummy_input.cpu())

output_dir = f"/teamspace/studios/this_studio/aimet/exported_models/bn_folded_fp16_embdedded"
os.makedirs(output_dir, exist_ok=True)
try:
    quant_sim.export(path=output_dir,
                filename_prefix="rtm_det",
                dummy_input=dummy_input.cuda(),
                use_embedded_encodings=True,
                export_to_torchscript=False)
except:
    shutil.rmtree(output_dir, ignore_errors=True)
    traceback.print_exc()


# In[67]:


model.model.eval()
m.eval()


# In[125]:


from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms

m_replaced = replace_rtm_bn(model)
m = prepare_model(m_replaced)
bn_pairs = fold_all_batch_norms(m, input_shapes=(1, 3, 640, 640))
print(len(bn_pairs))
m_replaced.eval()
m.eval()


# In[126]:


import torch.nn as nn

conv_module = nn.Sequential(*list(dict(m_replaced.named_modules())['backbone.stem.0'].children()))[:-1]
conv_module.eval()
print(conv_module)

with torch.no_grad():
    out = conv_module.cuda()(dummy_input.cuda())


# In[127]:


conv_module_bn = nn.Sequential(*list(dict(m.named_modules())['backbone.stem.0'].children()))[:-1]
conv_module_bn.eval()
print(conv_module_bn)

with torch.no_grad():
    bn_output = conv_module_bn.cuda()(dummy_input.cuda())


# In[128]:


# check if 2 matrices are equal
(out == bn_output).all()


# In[129]:


# calc diff between 2 matrices
(out - bn_output).abs()


# In[ ]:




