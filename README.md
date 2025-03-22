# Install

```bash
./setup.sh
# If you want to create a conda environment
./setup.sh --create-env
# If you want to use nsys profiling
./setup.sh --install-nsys
```


# Run

```bash
export CUDA_VISIBLE_DEVICES=0
for bs in 4 8 16 32; do
    bash train_sdxl_DDP.sh \
        --steps 100 \
        --micro-batch-size 2 \
        --batch-size $bs
done
export CUDA_VISIBLE_DEVICES=0,1
for bs in 4 8 16 32; do
    bash train_sdxl_DDP.sh \
        --steps 100 \
        --micro-batch-size 2 \
        --batch-size $bs
done        
```
