#!/bin/bash

pip install -U pip
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM && pip install -r requirements.txt
cd ..


python -c "import tensorrt_llm"
