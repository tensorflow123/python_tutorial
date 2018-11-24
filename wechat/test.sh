#!/bin/bash


for i in 0.003 0.01 0.03 0.1 0.3 1
do
    git co .
    sed -i 's/0.01/'"$i"'/' main.py
    sed -i 's/final_model.h5/final_model'"$i"'.h5/' main.py
    sed -i 's/loss_model.h5/loss_model'"$i"'.h5/' main.py

    python3 main.py
done

