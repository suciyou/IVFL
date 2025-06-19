#!/bin/bash

run_name="first"

for ((i = 0; i < 9; i++))
do
    python fscil.py mi $run_name $i
done

for ((i = 0; i < 9; i++))
do
    python fscil.py ci $run_name $i
done


for ((i =0; i < 11; i++))
do
    python fscil.py cu $run_name $i
done


python fscil_stat.py $run_name
