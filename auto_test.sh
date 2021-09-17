#!/bin/bash

target1="/path/to/VGG_2C"
echo "Hello"
let count=0
for f in "$target1"/model*
do
    echo $(basename $f)
    echo $f
    python new_evaluation.py $f
    #mv out_$(basename $f) th_50_vgg_train/
    let count=count+1
done
echo ""
echo "Count: $count"
