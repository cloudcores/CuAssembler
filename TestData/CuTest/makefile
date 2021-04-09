
INCLUDE="C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\ucrt"
LINK=-L "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64" -L "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"

ARCH50=-gencode=arch=compute_50,code=\"sm_50,compute_50\"
ARCH61=-gencode=arch=compute_61,code=\"sm_61,compute_61\"
ARCH75=-gencode=arch=compute_75,code=\"sm_75,compute_75\"
ARCH86=-gencode=arch=compute_86,code=\"sm_86,compute_86\"

cudatest.exe:cudatest.cu 
	nvcc cudatest.cu ${ARCH50} ${ARCH61} ${ARCH75} ${ARCH86} -o cudatest.exe -I ${INCLUDE} ${LINK} 

dsass:cudatest.exe
	cuobjdump -sass -arch sm_86 cudatest.exe > cudatest.sm_86.sass
	cuobjdump -sass -arch sm_75 cudatest.exe > cudatest.sm_75.sass
	cuobjdump -sass -arch sm_61 cudatest.exe > cudatest.sm_61.sass
	cuobjdump -sass -arch sm_52 cudatest.exe > cudatest.sm_52.sass

dcubin:cudatest.exe
	cuobjdump -xelf all cudatest.exe

dptx:cudatest.exe
	cuobjdump -ptx cudatest.exe > cudatest.ptx

