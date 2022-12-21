
.PHONY: clean

OUT = lib
BDIR = build
SDIR = src
CDIR = comp

CC_CUDA = nvcc
CC_LINUX = g++
CC_WIN = x86_64-w64-mingw32-g++ -static-libgcc -static-libstdc++
INC = -Iinc -arch=sm_52 -keep-dir $(CDIR)

_OBJS = rt.o

OBJS = $(patsubst %,$(BDIR)/%,$(_OBJS))

$(BDIR)/%.o: $(SDIR)/%.cu
	$(CC_CUDA) -c $(INC) -o $@ $< $(CFLAGS)

$(OUT): $(OBJS)
	$(CC_CUDA) $(SDIR)/rt.cu -o $(OUT)/rt -ccbin $(CC_LINUX)
	$(CC_CUDA) $(SDIR)/rt.cpp -o $(OUT)/rt.exe -ccbin $(CC_WIN)

$(SDIR)/%.h:
	echo test

clean:
	rm -f $(BDIR)/*.o $(OUT)/*

rebuild:
	make clean
	make

# app: rt.o
# 	g++ -o build/rt rt.cpp
# 	x86_64-w64-mingw32-g++ rt.cpp -o build/rt.exe -static-libgcc -static-libstdc++

# rt.o: rt.cpp
# 	g++ -c rt.cpp

