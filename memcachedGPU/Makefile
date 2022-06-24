
NVCC     ?= nvcc -c 
LINK     ?= nvcc

INCS := -I ../baseline/pr-stm -I .
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
	keyGenAndCacheStore.o \
#

# todo: uniform
all: memcdV1 memcdV2

memcdV1: memcdV1.o $(SHARED_OBJS)
	$(LINK) -o $@ $^

memcdV2: memcdV2.o $(SHARED_OBJS)
	$(LINK) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) -c -O3 -o $@ $^

clean:
	rm *.o ../src/*.o memcdV1 memcdV2