#!/bin/bash

### Usage: ./run.sh <action> <data> <disparity>
### <action>: all, exe, vis
### <data>: a/r/s (all/real/synthetic)
### <disparity>: non-negative integer

if [ $1 != 'all' ] && [ $1 != 'exe' ] && [ $1 != 'vis' ]; then
  echo -e "The action < $1 > is not available.(all/exe/vis)"
  exit
fi

if [ $2 != 'a' ] && [ $2 != 'r' ] && [ $2 != 's' ]; then
  echo -e "The data < $2 > is not available.(a/r/s)"
  exit
fi 

log_dir='./logs.txt'
if [ $1 == 'exe' ] || [ $1 == 'all' ]; then
  mkdir -p ./output/
  rm -r ./output/* 2> /dev/null
  if [ $2 == 'a' ] || [ $2 == 's' ]; then
    rm $log_dir 2> /dev/null
    for id in {0..9}; do
      echo -n "[Synthetic] $id : "
      log=$(python3 main.py -il=./data/Synthetic/TL$id.png -ir=./data/Synthetic/TR$id.png -o=./output/Synthetic_TL$id.pfm -gt=./data/Synthetic/TLD$id.pfm -d=$3)
      echo -n $log | rev | cut -d' ' -f1 | rev
      echo $log | rev | cut -d' ' -f1 | rev >> $log_dir
    done
    echo -e "\n======\nAverage Error: "
    python3 cal_avg.py $log_dir
    echo -e "======\n"
  fi
  if [ $2 == 'a' ] || [ $2 == 'r' ]; then
    for id in {0..9}; do
      echo -e "[Real] $id"
      python3 main.py -il=./data/Real/TL$id.bmp -ir=./data/Real/TR$id.bmp -o=./output/Real_TL$id.pfm -d=$3
    done
  fi
fi
if [ $1 == 'vis' ] || [ $1 == 'all' ]; then
  mkdir -p ./vis/
  rm -r ./vis/* 2> /dev/null
  if [ $2 == 'a' ] || [ $2 == 's' ]; then
    for id in {0..9}; do
      python3 visualize.py ./output/Synthetic_TL$id.pfm ./vis/Synthetic_TL$id.png
      python3 visualize.py ./data/Synthetic/TLD$id.pfm ./vis/Synthetic_TL${id}_D.png
    done
  fi
  if [ $2 == 'a' ] || [ $2 == 'r' ]; then
    for id in {0..9}; do
      python3 visualize.py ./output/Real_TL$id.pfm ./vis/Real_TL$id.png
    done
  fi
fi
