#!/bin/bash

# CPU model
echo -n "CPU model: "
lscpu | grep "Model name" | sed 's/.*: *//'

# Memory size
echo -n "Memory size: "
free -h | grep "Mem:" | awk '{print $2}'

# Memory type
echo -n "Memory type: "
dmidecode -t memory | grep -m 1 "Type:" | awk '{print $2}'

# Memory bandwidth (requires dmidecode with root privileges)
echo -n "RAM memory bandwidth: "
dmidecode -t memory | grep -m 1 "Speed:" | awk '{print $2 "Mbps"}'

# GPU connection
echo -n "GPU connection: "
lspci | grep -i nvidia | head -1 | grep -o "PCIe [0-9]\.[0-9] x[0-9]*" || echo "N/A"

# GPU topology
echo -n "GPU topology: "
nvidia-smi topo -m | grep "GPU" | head -1 | awk '{print $NF}' || echo "N/A"
