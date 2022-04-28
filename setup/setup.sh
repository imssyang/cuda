#!/bin/bash

APP=cuda
HOME=/opt/$APP
VER=v11.3
PIP=/opt/python3/bin/pip3

_create_symlink() {
  src=$1
  dst=$2
  if [[ ! -d $dst ]] && [[ ! -s $dst ]]; then
    ln -s $src $dst
    echo "($APP) create symlink: $src -> $dst"
  fi
}

_delete_symlink() {
  dst=$1
  if [[ -d $dst ]] || [[ -s $dst ]]; then
    rm -rf $dst
    echo "($APP) delete symlink: $dst"
  fi
}

_archive_item() {
  fpath=$1
  fdir=$(dirname $fpath)
  fname=$(basename $fpath)
  if [[ ! -f $HOME/archive/$fname-$VER.7z ]]; then
    echo "($APP) archive $fpath"
    7z a -mx9 -mmt4 $HOME/archive/$fname-$VER.7z $fpath
  fi
}

_unarchive_item() {
  fpath=$1
  fdir=$(dirname $fpath)
  fname=$(basename $fpath)
  if [[ ! -f $fpath ]]; then
    echo "($APP) unarchive $fpath"
    7z x $HOME/archive/$fname-$VER.txz -o$fdir
  fi
}

_large_files=(
  # cuda
  $HOME/$VER/lib64/libcublasLt.so.11.5.1.109
  $HOME/$VER/lib64/libcublasLt_static.a
  $HOME/$VER/lib64/libcublas.so.11.5.1.109
  $HOME/$VER/lib64/libcublas_static.a
  $HOME/$VER/lib64/libcufft.so.10.4.2.109
  $HOME/$VER/lib64/libcufft_static.a
  $HOME/$VER/lib64/libcufft_static_nocallback.a
  $HOME/$VER/lib64/libcufftw.so.10.4.2.109
  $HOME/$VER/lib64/libcusolverMg.so.11.1.2.109
  $HOME/$VER/lib64/libcusolver.so.11.1.2.109
  $HOME/$VER/lib64/libcusolver_static.a
  $HOME/$VER/lib64/libcusparse.so.11.6.0.109
  $HOME/$VER/lib64/libcusparse_static.a
  $HOME/$VER/lib64/libnvblas.so.11.5.1.109

  # cudnn
  $HOME/$VER/lib64/libcudnn_adv_infer.so.8.2.1
  $HOME/$VER/lib64/libcudnn_cnn_infer.so.8.2.1
  $HOME/$VER/lib64/libcudnn_cnn_infer_static.a
  $HOME/$VER/lib64/libcudnn_cnn_train.so.8.2.1
  $HOME/$VER/lib64/libcudnn_cnn_train_static.a
  $HOME/$VER/lib64/libcudnn_ops_infer.so.8.2.1
  $HOME/$VER/lib64/libcudnn_static.a

  # TensorRT
  $HOME/$VER/lib64/libnvinfer.so.8.0.3
  $HOME/$VER/lib64/libnvinfer_static.a
)

init() {
  _create_symlink $HOME/$VER            /usr/local/cuda
  _create_symlink $HOME/$VER/bin        $HOME/bin
  _create_symlink $HOME/$VER/include    $HOME/include
  _create_symlink $HOME/$VER/lib64      $HOME/lib64

  chown -R root:root $HOME
  chmod 755 $HOME

  . $HOME/setup/env.sh
  $PIP install --no-cache-dir -r $HOME/setup/requirements.txt
}

deinit() {
  _delete_symlink /usr/local/cuda
  _delete_symlink $HOME/bin
  _delete_symlink $HOME/include
  _delete_symlink $HOME/lib64

  $PIP uninstall --no-cache-dir -r $HOME/setup/requirements.txt
}

archive() {
  for fpath in "${_large_files[@]}"; do
    _archive_item $fpath
  done
}

unarchive() {
  for fpath in "${_large_files[@]}"; do
    _unarchive_item $fpath
  done
}

show() {
  for fpath in "${_large_files[@]}"; do
    echo $fpath
  done
}

case "$1" in
  init) init ;;
  deinit) deinit ;;
  archive) archive ;;
  unarchive) unarchive ;;
  show) show ;;
  *) SCRIPTNAME="${0##*/}"
     echo "Usage: $SCRIPTNAME {init|deinit|archive|unarchive|show}"
     exit 3
     ;;
esac

exit 0

# vim: syntax=sh ts=4 sw=4 sts=4 sr noet
