
cd aimet_torch
cp activation.py /usr/local/lib/python3.10/dist-packages/aimet_torch/transformers/ 
cp auto_quant.py /usr/local/lib/python3.10/dist-packages/aimet_torch/
cp batch_norm_fold.py /usr/local/lib/python3.10/dist-packages/aimet_torch/
cp model_preparer.py /usr/local/lib/python3.10/dist-packages/aimet_torch/
cp qc_quantize_op.py /usr/local/lib/python3.10/dist-packages/aimet_torch/
cp quant_analyzer.py /usr/local/lib/python3.10/dist-packages/aimet_torch/
cp quantsim.py /usr/local/lib/python3.10/dist-packages/aimet_torch/

echo "Aimet files copied"

cd ../torch
cp activation.py /usr/local/lib/python3.10/dist-packages/torch/nn/modules
cp functional.py /usr/local/lib/python3.10/dist-packages/torch/nn/

echo "Torch files copied"