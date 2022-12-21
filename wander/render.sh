echo -e "\e[32m--- Compiling ---\e[0m"
make rebuild

echo -e "\n\e[36m--- Rendering Image ---\e[0m"
nvprof ./lib/wander --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer