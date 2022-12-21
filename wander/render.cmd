@echo off
cls

echo [32m--- Compiling ---[0m
make rebuild

echo [36m--- Rendering Image ---[0m
nvprof .\lib\wander.exe --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer