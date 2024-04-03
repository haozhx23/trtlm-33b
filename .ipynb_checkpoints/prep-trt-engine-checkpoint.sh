#!/bin/bash

base_path=/home/ec2-user/SageMaker
# base_path=/workspace
# hf_model_path=${base_path}/chinese-llama-33b
# trt_ckpt_path=${base_path}/trt-lm-converted-ckpts/chinese_llama33b_int8_ckpt
# trt_engine_path=${base_path}/trt-lm-built-engines/chinese_llama33b_int8_engine


hf_model_path=${base_path}/deepseek-coder-33b
trt_ckpt_path=${base_path}/trt-lm-converted-ckpts/deepseek-coder-33b-int8-ckpt
trt_engine_path=${base_path}/trt-lm-built-engines/deepseek-coder-33b-int8-engine

        # --int8_kv_cache \
python TensorRT-LLM/examples/llama/convert_checkpoint.py \
        --model_dir ${hf_model_path}/ \
        --output_dir ${trt_ckpt_path}/ \
        --dtype float16 \
        --use_weight_only \
        --weight_only_precision int8 \
        --tp_size 2



bs=4
ilen=538
olen=152

trtllm-build --checkpoint_dir ${trt_ckpt_path}/ \
            --output_dir ${trt_engine_path}_${bs}_${ilen}_${olen}/ \
            --gemm_plugin float16 \
            --use_custom_all_reduce disable \
            --max_batch_size ${bs} \
            --max_input_len ${ilen} \
            --max_output_len ${olen}
            
            