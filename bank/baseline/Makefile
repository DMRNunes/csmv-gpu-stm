
NVCC     ?= nvcc -c 
LINK     ?= nvcc

INCS := #-I ../src -I .
LIBS :=

NVFLAGS := -rdc=true --maxrregcount 63
LDFLAGS := -rdc=true --maxrregcount 63

### use "$ make DEBUG=1"
ifeq ($(DEBUG),1)
NVFLAGS += -g -G
endif

NVFLAGS += $(INCS)
LDFLAGS += $(LIBS)

SHARED_OBJS := \
	API.o 	\
	#../src/API.o \
#

all: uniform sliced

uniform: kernel.o $(SHARED_OBJS)
	$(LINK) -o $@ $^

sliced: disjoint.o $(SHARED_OBJS)
	$(LINK) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm *.o ../src/*.o uniform sliced