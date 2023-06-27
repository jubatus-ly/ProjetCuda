# Makefile pour compiler tous les fichiers CUDA et C++ avec leurs commandes respectives

CXX := g++
NVCC := nvcc
CXXFLAGS := $(shell pkg-config --cflags opencv)
LDFLAGS := $(shell pkg-config --libs opencv)

CU_SRCS := $(wildcard *.cu)
CU_EXES := $(patsubst %.cu,%,$(CU_SRCS))

CPP_SRCS := $(wildcard *.cpp)
CPP_EXES := $(patsubst %.cpp,%,$(CPP_SRCS))

.PHONY: all clean

all: $(CU_EXES) $(CPP_EXES)

%: %.cu
	$(NVCC) -o $@ $< -std=c++11 $(LDFLAGS)
	chmod a+x $@

%: %.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)
	chmod a+x $@

clean:
	rm -f $(CU_EXES) $(CPP_EXES)