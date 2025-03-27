#! /bin/bash

pip install compress_pickle torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu

original_model_path=${ORIGINAL_MODEL_PATH:-/models/DeepSeek-R1}
output_model_root=${OUTPUT_MODEL_ROOT:-/models}
input_scales_path=DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz

device=$(hl-smi -Q name -f csv | tail -n 1)

# checke if "HL-225" is in the device name
if [[ $device == *"HL-225"* ]]; then
    echo "Converting weights for Gaudi2"
    target="G2"
    full_range=240.0    # torch.finfo(torch.float8_e4m3fnuz).max for Gaudi2
# check if "HL-328" is in the device name
elif [[ $device == *"HL-328"* ]]; then
    echo "Converting weights for Gaudi3"
    target="G3"
    full_range=448.0    # torch.finfo(torch.float8_e4m3fn).max for Gaudi3
else
    echo "Unknown device: $device"
    exit 1
fi

dynamic_model_path=${output_model_root}/DeepSeek-R1-${target}-dynamic
static_model_path=${output_model_root}/DeepSeek-R1-${target}-static

workdir=`pwd`

echo "Converting weights and scales in *.safetensors and update model.safetensors.index.json"
echo "Detailed log is in convert_block_fp8_to_channel_fp8.log"
python convert_block_fp8_to_channel_fp8.py \
    --model_path $original_model_path \
    --qmodel_path $dynamic_model_path \
    --full_range $full_range \
    --input_scales_path $input_scales_path \
    > convert_block_fp8_to_channel_fp8.log 2>&1

echo "Copy extral model files"
mkdir -p $dynamic_model_path
cp -rLn $original_model_path/* $dynamic_model_path
# use the config.json for dynamic activation quant
cp $workdir/config_dynamic.json $dynamic_model_path/config.json

mkdir -p $static_model_path
cd $static_model_path
echo "Re-use the dynamic quant weights by soft link"
ln -s ../DeepSeek-R1-${target}-dynamic/* .
# use the config.json for static activation quant
cp --remove-destination $workdir/config_static.json config.json
