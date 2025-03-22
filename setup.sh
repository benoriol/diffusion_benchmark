## If you want to use nsys profiling
INSTALL_NSYS=false
CREATE_ENV=false

# Parse command line arguments
for arg in "$@"; do
    if [ "$arg" = "--install-nsys" ]; then
        INSTALL_NSYS=true
    fi
    if [ "$arg" = "--create-env" ]; then
        CREATE_ENV=true
    fi
done

# Install nsys if requested
if [ "$INSTALL_NSYS" = true ]; then
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
    sudo apt install nsight-systems
fi

# Create environment if requested
if [ "$CREATE_ENV" = true ]; then
    conda create -n diffbench python=3.11 -y
    conda activate diffbench
fi

# Install this, maybe create a conda first
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install diffusers["torch"] transformers
pip install bitsandbytes
pip install hf_transfer

# THis will download the model
export CUDA_VISIBLE_DEVICES=0; bash train_sdxl_DDP.sh --steps 1 --micro-batch-size 1 --batch-size 1