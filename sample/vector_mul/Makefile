CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS     := -m64
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
endif
CCFLAGS       :=
LDFLAGS       :=
INCLUDES      := #-I./Common
LIBRARIES     :=

ALL_CCFLAGS   := --threads 0 --std=c++11
ALL_CCFLAGS   += $(NVCCFLAGS)
ALL_CCFLAGS   += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS   :=
ALL_LDFLAGS   += $(NVCCFLAGS)
ALL_LDFLAGS   += $(addprefix -Xlinker ,$(LDFLAGS))

ifeq ($(GENCODE_FLAGS),)
    SMS ?= 52 60 61 70 75 80 86
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

all: matrix_mul

matrix_mul: matrix_mul.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) $+ -o $@

matrix_mul.o: matrix_mul.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -c $< -o $@

run:
	./matrix_mul

clean:
	rm -f matrix_mul matrix_mul.o
