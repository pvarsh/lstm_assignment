#!/bin/bash
wget https://dl.dropboxusercontent.com/u/12075991/char_thirteen_epoch_model_seq50.net
luajit main.lua -seq_length 50 -no_train $true -load $true -load_name char_thirteen_epoch_model_seq50.net -submission $true -char $true
