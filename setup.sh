sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install diffusers["torch"] transformers
pip install bitsandbytes

export CUDA_VISIBLE_DEVICES=0; bash train_sdxl_DDP.sh --steps 1 --micro-batch-size 1 --batch-size 1