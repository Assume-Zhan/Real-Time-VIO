# ML-Term-Project
Machine Learning Term Projject

## Background for Visual Odometry

| Topics | Description | Reference Links | Other |
|--------------|-------------|--------------------------|-------|
| Dataset | KITTI dataset | [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) | - |

## Run in Docker

### Pre-requisites

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [Nvidia-Docker](http://web.archive.org/web/20230627162323/https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Build and Run the image

```bash
docker-compose up --build -d 
```

### Attach to the container

```bash
docker attach pytorch-vio
```

### Run with devcontainer

- Following below link
[https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers)
