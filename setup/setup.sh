#!/bin/bash

APP=cuda
HOME=/opt/$APP

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

init() {
  _create_symlink $HOME/v11.4/bin        $HOME/bin
  _create_symlink $HOME/v11.4/include    $HOME/include
  _create_symlink $HOME/v11.4/lib64      $HOME/lib64

  chown -R root:root $HOME
  chmod 755 $HOME
}

deinit() {
  _delete_symlink $HOME/bin
  _delete_symlink $HOME/include
  _delete_symlink $HOME/lib64
}

case "$1" in
  init) init ;;
  deinit) deinit ;;
  *) SCRIPTNAME="${0##*/}"
     echo "Usage: $SCRIPTNAME {init|deinit}"
     exit 3
     ;;
esac

exit 0

# vim: syntax=sh ts=4 sw=4 sts=4 sr noet
