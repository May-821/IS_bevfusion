## Usage

### Prerequisites

The code is built with following libraries:

- CUDA = **11.3**
- gcc version = 9.4 (versions that are too new may cause compatibility issues.)
- Python >= **3.8**, \<3.9
  ```bash
  conda create -n {env_name} python=3.8 -y
  ```
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
  ```bash
  pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```
- [torchpack](https://github.com/mit-han-lab/torchpack)
  ```bash
  pip install torchpack==0.3.1
  ```
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
  - download OpenMPI source code:
  ```bash
  wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz
  tar xf ./openmpi-4.0.4.tar.gz
  ```
  - set compilation parameters:
  ```bash
  cd openmpi-4.0.4
  ./configure --prefix=<path>
  ```
  ```--prefix``` is used to specify the installation path for OpenMPI, where the compiled files will be placed after completion.
  - compile:
  ```bash
  make
  # or
  make -j4 # indicates compiling with four threads.
  ```
  - installation:
  ```bash
  make install
  ```
  - environment variable settings:
  
  add these two lines to ~/.bashrc
  ```bash
  export PATH=/opt/openmpi/bin:$PATH
  export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
  ```

  Then you can start to install mpi4py:
  ```bash
  pip install mpi4py==3.0.3
  ```
  
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
  ```bash
  pip install mmcv-full==1.4.0
  ```
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
  ```bash
  pip install mmdet==2.20.0
  ```
- TorchEx
  ```bash
  cd ./mmdet3d/ops/TorchEx
  pip install -v .
  ```
<!-- - Pillow = **8.4.0** (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
  ```bash
  pip install pillow==8.4.0
  ``` -->
<!-- - [tqdm](https://github.com/tqdm/tqdm) -->
<!-- - [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
  ```bash
  pip install nuscenes-devkit
  ``` -->
After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```
Then, run this command to install the the remaining packages:
```bash
pip install -r requirements.txt
```

### Data Preparation

#### nuScenes

You can run this command to download the full NuScenes dataset:
```bash
bash ./data/dataset.sh
```

Then run this command to preprocess:
```bash
python ./tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

After data preparation, you will be able to see the following directory structure:

```
IS_bevfusion
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_gt_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Training

Please download the checkpoints using the following script: 

```bash
./tools/download_pretrained.sh
```
We provide instructions to reproduce our results on nuScenes.
you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/train.py [config file path] --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/[checkpoint name].pth
```

For example, if you want to train the camera-only variant for object detection after modified, please run:

```bash
torchpack dist-run -np 2 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default_V2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.

### Evaluation

We also provide instructions for evaluating our pretrained models. 

Then, you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/test.py [config file path] [checkpoint path] --eval [evaluation type]
```

For example, if you want to evaluate the camera-only variant for object detection after modified, you can try:

```bash
torchpack dist-run -np 2 python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default_V2.yaml [checkpoint path] --eval bbox
```

