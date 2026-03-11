## Installation Guide

1. **Get Project**
   ```bash
   git clone https://github.com/BenjaminPang/outfit-hub.git
   cd outfit_hub

2. **Create Conda Env**
    ```bash
    conda create -n outfit_hub python=3.10 -y
    conda activate outfit_hub

3. **Install Right Pytorch**
    
    Depending on your CUDA version, obtain the installation command from the PyTorch website.
    ```bash
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
    pip install -r requirements.txt

4. **Install Project**

    Install the project in editable mode
    ```bash
    pip install -e .