
BASENAME=l2bank_test

INCLUDE=-I "C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\ucrt" -I "../../include"
LINK=-L "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64" -L "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"

ARCH50=-gencode=arch=compute_50,code=\"sm_50,compute_50\"
ARCH61=-gencode=arch=compute_61,code=\"sm_61,compute_61\"
ARCH75=-gencode=arch=compute_75,code=\"sm_75,compute_75\"
ARCH86=-gencode=arch=compute_86,code=\"sm_86,compute_86\"

${BASENAME}.exe:${BASENAME}.cu 
	nvcc ${BASENAME}.cu ${ARCH50} ${ARCH75} -o ${BASENAME}.exe ${INCLUDE} ${LINK} 

dsass:${BASENAME}.exe
	cuobjdump -sass -arch sm_75 ${BASENAME}.exe > ${BASENAME}.sm_75.sass
	cuobjdump -sass -arch sm_50 ${BASENAME}.exe > ${BASENAME}.sm_50.sass

dcubin:${BASENAME}.exe
	cuobjdump -xelf all ${BASENAME}.exe

dptx:${BASENAME}.exe
	cuobjdump -ptx ${BASENAME}.exe > ${BASENAME}.ptx

