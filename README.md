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
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
sh cuda_11.4.4_470.82.01_linux.run --installpath=/opt/cuda/v11.4

   安装内容：
   /etc/X11/xorg.conf                     系统默认使用NVIDIA GPU (手动)
   /usr/local/cuda                        软链接（指向最新CUDA Toolkit）
   /usr/local/cuda-11.4                   CUDA Toolkit
   $(HOME)shell/NVIDIA_CUDA-11.4_Samples  CUDA Samples
   /dev/nvidia* (0666权限)                CUDA与GPU通信使用

环境配置
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


卸载
/usr/local/cuda/bin/cuda-uninstaller      卸载CUDA
/usr/bin/nvidia-uninstall                 卸载驱动
```

# 安装cudnn

```shell
echo
```

# python环境

```shell
python3 -m pip install --upgrade setuptools pip wheel  更新包工具
python3 -m pip install nvidia-pyindex    支持从NVIDIA NGC PyPI repo获取模块
requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

python3 -m pip install nvidia-cuda-runtime-cu11  CUDA运行时包
python3 -m pip install nvidia-<library>
```

