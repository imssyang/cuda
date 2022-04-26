# 安装驱动

启用`--gpus all`后容器与宿主机共用驱动，因此宿主机安装驱动即可。

```shell
uname -m && cat /etc/*release   查看OS信息
lspci | grep -i nvidia          查看GPU硬件
lsmod | grep nouveau                       查看nouveau驱动
vi /etc/modprobe.d/blacklist-nouveau.conf  禁用nouveau驱动
  blacklist nouveau
  options nouveau modeset=0
```

# 安装CUDA

```shell
# 交互式命令安装
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sh cuda_11.3.1_465.19.01_linux.run --installpath=/opt/cuda/v11.3  (not install driver)

   安装内容：
   /etc/X11/xorg.conf                     系统默认使用NVIDIA GPU (手动)
   /usr/local/cuda                        软链接（指向最新CUDA Toolkit）
   /usr/local/cuda-11.3                   CUDA Toolkit
   $(HOME)shell/NVIDIA_CUDA-11.3_Samples  CUDA Samples
   /dev/nvidia* (0666权限)                CUDA与GPU通信使用

环境配置
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


卸载
/usr/local/cuda/bin/cuda-uninstaller      卸载CUDA
/usr/bin/nvidia-uninstall                 卸载驱动
```

## 独立的python-cuda运行时环境

```shell
python3 -m pip install --upgrade setuptools pip wheel  更新包工具
python3 -m pip install nvidia-pyindex    支持从NVIDIA NGC PyPI repo获取模块
requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

python3 -m pip install nvidia-cuda-runtime-cu11  CUDA运行时包
python3 -m pip install nvidia-<library>
    nvidia-nvml-dev-cu114
    nvidia-cuda-nvcc-cu114
    nvidia-cuda-runtime-cu114
    nvidia-cuda-cupti-cu114
    nvidia-cublas-cu114
    nvidia-cuda-sanitizer-api-cu114
    nvidia-nvtx-cu114
    nvidia-cuda-nvrtc-cu114
    nvidia-npp-cu114
    nvidia-cusparse-cu114
    nvidia-cusolver-cu114
    nvidia-curand-cu114
    nvidia-cufft-cu114
    nvidia-nvjpeg-cu114
```

# 安装cudnn

```shell
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -xvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
cp cuda/include/cudnn*.h /opt/cuda/v11.3/include
cp -P cuda/lib64/libcudnn* /opt/cuda/v11.3/lib64
chmod a+r /opt/cuda/v11.3/include/cudnn*.h /opt/cuda/v11.3/lib64/libcudnn*
```

# 安装TensorRT
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0.3/tars/tensorrt-8.0.3.4.linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz
mv TensorRT-8.0.3.4/bin/*        /opt/cuda/v11.3/bin/
mv TensorRT-8.0.3.4/include/*    /opt/cuda/v11.3/include/
mv TensorRT-8.0.3.4/lib/stubs/*  /opt/cuda/v11.3/lib64/stubs
mv TensorRT-8.0.3.4/lib/*        /opt/cuda/v11.3/lib64
cp TensorRT-8.0.3.4/python/tensorrt-8.0.3.4-cp38-none-linux_x86_64.whl              /opt/cuda/setup/tensorrt
cp TensorRT-8.0.3.4/uff/uff-0.6.9-py2.py3-none-any.whl                              /opt/cuda/setup/tensorrt
cp TensorRT-8.0.3.4/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl            /opt/cuda/setup/tensorrt
cp TensorRT-8.0.3.4/onnx_graphsurgeon/onnx_graphsurgeon-0.3.10-py2.py3-none-any.whl /opt/cuda/setup/tensorrt

# 安装PyTorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

