BASENAME=cudatest
BINNAME=${BASENAME}

INCLUDE=
LINK=

ARCH60=-gencode=arch=compute_60,code=\"sm_60,compute_60\"
ARCH61=-gencode=arch=compute_61,code=\"sm_61,compute_61\"
ARCH70=-gencode=arch=compute_70,code=\"sm_70,compute_70\"
ARCH75=-gencode=arch=compute_75,code=\"sm_75,compute_75\"
ARCH80=-gencode=arch=compute_80,code=\"sm_80,compute_80\"
ARCH86=-gencode=arch=compute_86,code=\"sm_86,compute_86\"

ARCH=sm_75

BUILD_ARGS=${BASENAME}.cu -arch=${ARCH} -o ${BINNAME} ${INCLUDE} ${LINK}

${BINNAME}:${BASENAME}.cu 
	nvcc ${BUILD_ARGS}

dump:${BASENAME}.cu
	HNVCC_OP=dump hnvcc ${BUILD_ARGS}

d2h:dump
	cuasm dump.${BASENAME}.${ARCH}.cubin
	cp -b dump.${BASENAME}.${ARCH}.cuasm hack.${BASENAME}.${ARCH}.cuasm

hack:${BASENAME}.cu
	cuasm hack.${BASENAME}.${ARCH}.cuasm
	HNVCC_OP=hack hnvcc ${BUILD_ARGS}

dcubin:${BINNAME}
	cuobjdump -xelf all ${BASENAME}

dptx:${BINNAME}
	cuobjdump -ptx ${BASENAME} > ${BASENAME}.ptx

dsass:${BINNAME}
	cuobjdump -sass ${BASENAME} > ${BASENAME}.sass

.PHONY: clean

clean:
	-rm ${BINNAME}