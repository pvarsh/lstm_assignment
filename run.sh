#!/bin/bash
luajit main.lua -seq_length 50 -no_train $true -load $true -load_name char_thirteen_epoch_model_seq50.net -submission $true -char $true
