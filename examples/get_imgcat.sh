#!/bin/sh
pip install --user imgcat
mv $HOME/.local/bin/imgcat $HOME/bin/imgcat
export PATH=$HOME/bin/:$PATH
