
OUT = lib
BDIR = build
SDIR= src

CC_LINUX = g++
CC_WIN = x86_64-w64-mingw32-g++ -static-libgcc -static-libstdc++
INC = -Iinc

_OBJS = rt.o

OBJS = $(patsubst %,$(BDIR)/%,$(_OBJS))

$(BDIR)/%.o: $(SDIR)/%.cpp
	$(CC_LINUX) -c $(INC) -o $@ $< $(CFLAGS)

$(OUT): $(OBJS)
	$(CC_LINUX) $(SDIR)/rt.cpp -o $(OUT)/rt
	$(CC_WIN) $(SDIR)/rt.cpp -o $(OUT)/rt


clean:
	rm -f $(BDIR)/*.o $(OUT)


# app: rt.o
# 	g++ -o build/rt rt.cpp
# 	x86_64-w64-mingw32-g++ rt.cpp -o build/rt.exe -static-libgcc -static-libstdc++

# rt.o: rt.cpp
# 	g++ -c rt.cpp

