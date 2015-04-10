libGPUArray: GPUArray.c
	nvcc -o libGPUArray.so --shared -lcublas --compiler-options '-fPIC' GPUArray.c -I/home/magania/julia/include
#	gcc -o libGPUArray.so --share -fPIC GPUArray.o 