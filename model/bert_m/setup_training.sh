#!/bin/bash


data_dir=/content/drive/MyDrive/coref/models


download_bert(){
  model=$1
  wget -P $data_dir https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
  unzip $data_dir/$model.zip
  rm $data_dir/$model.zip
  mv $model $data_dir/
}
download_bert multilingual_L-12_H-768_A-12

#vocab_file=cased_config_vocab/vocab.txt
#python minimize.py $vocab_file $data_dir $data_dir false
