CC = gcc

ifeq ($(CC), gcc)
	CFLAGS = -W -Werror -Wall -Wextra -pedantic -Wno-pointer-arith -fPIC
else ifeq ($(CC), clang)
	CFLAGS = -W -Werror -Wall -Wextra -pedantic -Wno-gnu-pointer-arith -fPIC
endif

archives = wekua.o utils.o
modules = tensor

ifeq ($(MODE), debug)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fno-omit-frame-pointer
else ifeq ($(MODE), debug-nvidia)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fsanitize-recover=address
else
	DEBUG_FLAGS = -O2
endif

export CC
export CFLAGS
export DEBUG_FLAGS

main: $(modules) $(archives)
	$(CC) $(CFLAGS) -shared $(DEBUG_FLAGS) -lOpenCL -pthread build/*.o -o libwekua.so -lm

%.o: src/%.c
	$(CC) -c $(CFLAGS) $(DEBUG_FLAGS) $< -o build/$@

$(modules):
	$(MAKE) -C src/$@

install:
	cp libwekua.so /usr/lib/
	rm -rf /usr/include/wekua
	cp -r headers /usr/include/wekua
	rm -rf /usr/lib/wekua_kernels/
	cp -r kernels/ /usr/lib/wekua_kernels/
	chmod 755 /usr/lib/wekua_kernels
	chmod 644 /usr/lib/wekua_kernels/*
	chmod 755 /usr/include/wekua
	chmod 644 /usr/include/wekua/*

uninstall:
	rm -rf /usr/lib/libwekua.so
	rm -rf /usr/include/wekua
	rm -rf /usr/lib/wekua_kernels

clean:
	rm -rf build/*.o