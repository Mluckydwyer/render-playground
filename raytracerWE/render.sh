echo -e "\e[32m--- Compiling ---\e[0m"
make

echo -e "\n\e[36m--- Rendering Image ---\e[0m"
./lib/rt > image.ppm