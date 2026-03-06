#!/bin/bash
# ─── Install Nsight Systems (per NSYS_doc.pdf) ───
# Run this ONCE on HPC login node (NOT inside singularity)

# Download
cd ~
mkdir -p ~/installers && cd ~/installers
wget -O NsightSystems.run \
  https://developer.nvidia.com/downloads/assets/tools/secure/nsightsystems/2026_1/NsightSystems-linux-public-2026.1.1.204-3717666.run

# Make executable
chmod +x NsightSystems.run

# Extract to user directory
mkdir -p ~/tools/nsight-systems
./NsightSystems.run --target ~/tools/nsight-systems --noexec

# Add to PATH
NSYS_DIR=$(dirname $(find ~/tools/nsight-systems -type f -name nsys -perm -111 | head -n1))
echo "export PATH=$NSYS_DIR:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify
nsys --version
