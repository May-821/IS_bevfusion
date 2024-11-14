## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
  ```bash
  pip install mmcv-full==1.4.0
  ```
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
  ```bash
  pip install mmdet==2.20.0
  ```
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

if encounters these problem:
- `TypeError: FormatCode() got an unexpected keyword argument 'verify'`:
  Solution: pip install yapf==0.40.1
- 

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

We also provide a [Dockerfile](docker/Dockerfile) to ease environment setup. To get started with docker, please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t bevfusion
```

We can then run the docker with the following command:

```bash
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

We recommend the users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone our repo and install custom CUDA extensions:

```bash
cd home && git clone https://github.com/mit-han-lab/bevfusion && cd bevfusion
python setup.py develop
```

You can then create a symbolic link `data` to the `/dataset` directory in the docker.

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Evaluation

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script: 

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
```

While for the segmentation variant of BEVFusion, this command will be helpful:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the camera-only variant for object detection, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For camera-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For LiDAR-only detector, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

For LiDAR-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

For BEVFusion detection model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth 
```

For BEVFusion segmentation model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.