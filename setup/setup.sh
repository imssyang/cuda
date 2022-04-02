#!/bin/bash

APP=cuda
HOME=/opt/$APP
VER=v11.4

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
  XZ_OPT=-e9 tar -cJf $HOME/archive/$VER-$fname.txz -C $fdir $fname
}

_unarchive_item() {
  fpath=$1
  fdir=$(dirname $fpath)
  fname=$(basename $fpath)
  tar -xvf $HOME/archive/$VER-$fname.txz -C $fdir
}

init() {
  _create_symlink $HOME/$VER/bin        $HOME/bin
  _create_symlink $HOME/$VER/include    $HOME/include
  _create_symlink $HOME/$VER/lib64      $HOME/lib64

  chown -R root:root $HOME
  chmod 755 $HOME
}

deinit() {
  _delete_symlink $HOME/bin
  _delete_symlink $HOME/include
  _delete_symlink $HOME/lib64
}

archive() {
  _archive_item $HOME/$VER/lib64/libcublasLt.so.11.6.5.2
  _archive_item $HOME/$VER/lib64/libcublasLt_static.a
  _archive_item $HOME/$VER/lib64/libcublas.so.11.6.5.2
  _archive_item $HOME/$VER/lib64/libcublas_static.a
  _archive_item $HOME/$VER/lib64/libcufft.so.10.5.2.100
  _archive_item $HOME/$VER/lib64/libcufft_static.a
  _archive_item $HOME/$VER/lib64/libcufft_static_nocallback.a
  _archive_item $HOME/$VER/lib64/libcusolverMg.so.11.2.0.120
  _archive_item $HOME/$VER/lib64/libcusolver.so.11.2.0.120
  _archive_item $HOME/$VER/lib64/libcusolver_static.a
  _archive_item $HOME/$VER/lib64/libcusparse.so.11.6.0.120
  _archive_item $HOME/$VER/lib64/libcusparse_static.a
}

unarchive() {
  _unarchive_item $HOME/$VER/lib64/libcublasLt.so.11.6.5.2
  _unarchive_item $HOME/$VER/lib64/libcublasLt_static.a
  _unarchive_item $HOME/$VER/lib64/libcublas.so.11.6.5.2
  _unarchive_item $HOME/$VER/lib64/libcublas_static.a
  _unarchive_item $HOME/$VER/lib64/libcufft.so.10.5.2.100
  _unarchive_item $HOME/$VER/lib64/libcufft_static.a
  _unarchive_item $HOME/$VER/lib64/libcufft_static_nocallback.a
  _unarchive_item $HOME/$VER/lib64/libcusolverMg.so.11.2.0.120
  _unarchive_item $HOME/$VER/lib64/libcusolver.so.11.2.0.120
  _unarchive_item $HOME/$VER/lib64/libcusolver_static.a
  _unarchive_item $HOME/$VER/lib64/libcusparse.so.11.6.0.120
  _unarchive_item $HOME/$VER/lib64/libcusparse_static.a
}

case "$1" in
  init) init ;;
  deinit) deinit ;;
  archive) archive ;;
  unarchive) unarchive ;;
  *) SCRIPTNAME="${0##*/}"
     echo "Usage: $SCRIPTNAME {init|deinit|archive|unarchive}"
     exit 3
     ;;
esac

exit 0

# vim: syntax=sh ts=4 sw=4 sts=4 sr noet
