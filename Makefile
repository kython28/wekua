CC = gcc
archives = wekua.o matrix.o print.o trig.o blas.o extra.o aberth_root.o linear.o acti.o acti_linear.o acti_sigmoid.o acti_tanh.o acti_relu.o acti_leakyrelu.o werror.o network.o neuron.o optim.o file.o fifo.o regularization.o

COMMON_CFLAGS = -W -Werror -Wall -Wextra -Wpedantic -Wno-pointer-arith -fPIC -Wshadow -Wunused-result -Wno-unused-command-line-argument -Wconversion -Wformat=2 -Wformat-security -Wnull-dereference -Wuninitialized -Wstrict-aliasing=2 -Wundef -Wunreachable-code -Wfloat-equal -Wredundant-decls -Wwrite-strings -Wdouble-promotion -Wswitch-enum
ifeq ($(CC), gcc)
	CFLAGS = $(COMMON_CFLAGS) -Wlogical-op
else
	CFLAGS = $(COMMON_CFLAGS)
endif

ifeq ($(MODE), debug)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fno-omit-frame-pointer
else ifeq ($(MODE), debug-nvidia)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fsanitize-recover=address
else ifeq ($(MODE), debug-no-sanitize)
	DEBUG_FLAGS = -O0 -g
else ifeq ($(MODE), analyze)
	COMMON_CFLAGS = -fPIC -Werror
	ifeq ($(CC), gcc)
		DEBUG_FLAGS = -O0 -g -fanalyzer -Wno-analyzer-malloc-leak
	else
		DEBUG_FLAGS = -O0 -g -Wno-unused-command-line-argument --analyze --analyze -Xclang -analyzer-config -Xclang crosscheck-with-z3=true
	endif
else
	DEBUG_FLAGS = -O2 -D_FORTIFY_SOURCE=3
endif

main: $(archives)
	$(CC) $(CFLAGS) -shared $(DEBUG_FLAGS) -lOpenCL -pthread $(archives) -o libwekua.so -lm

%.o: old_src/%.c
	$(CC) -c $(CFLAGS) $(DEBUG_FLAGS) $< -o $@

install:
	cp libwekua.so /usr/lib/
	rm -rf /usr/include/wekua
	cp -r headers /usr/include/wekua
	rm -rf /usr/lib/wekua_kernels/
	cp -r old_kernels/ /usr/lib/wekua_kernels/
	chmod 755 /usr/lib/wekua_kernels
	chmod 644 /usr/lib/wekua_kernels/*
	chmod 755 /usr/include/wekua
	chmod 644 /usr/include/wekua/*

uninstall:
	rm -rf /usr/lib/libwekua.so
	rm -rf /usr/include/wekua
	rm -rf /usr/lib/wekua_kernels

clean:
	rm -rf $(archives)
