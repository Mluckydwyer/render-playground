
.PHONY: clean

OUT = lib
BDIR = build
SDIR = src
CDIR = comp

CC_CUDA = nvcc
CC_LINUX = g++
CC_WIN = 
INC = -Iinc -arch=sm_52 -keep-dir $(CDIR)

_OBJS = wander.o core/vec3.o shapes/sphere.o

OBJS = $(patsubst %,$(BDIR)/%,$(_OBJS))

$(BDIR)/%.o: $(SDIR)/%.cu
	$(CC_CUDA) -c $(INC) -o $@ $< $(CFLAGS)

$(OUT): $(OBJS)
ifeq ($(OS),Windows_NT)
	$(CC_CUDA) $(SDIR)/wander.cpp -o $(OUT)/wander.exe -Xcompiler "$(CC_WIN)"
else
	$(CC_CUDA) $(SDIR)/wander.cpp -o $(OUT)/wander -ccbin $(CC_LINUX)
endif

#C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64

clean:
ifeq ($(OS),Windows_NT)
	del /f /q $(OUT)\*
	del /f /q $(BDIR)\*.o
else
	rm -f $(BDIR)/*.o $(OUT)/*
endif


rebuild:
	make clean
	make

# app: rt.o
# 	g++ -o build/rt rt.cpp
# 	x86_64-w64-mingw32-g++ rt.cpp -o build/rt.exe -static-libgcc -static-libstdc++

# rt.o: rt.cpp
# 	g++ -c rt.cpp

