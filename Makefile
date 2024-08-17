CC = gcc
archives = wekua.o matrix.o print.o trig.o blas.o extra.o aberth_root.o linear.o acti.o acti_linear.o acti_sigmoid.o acti_tanh.o acti_relu.o acti_leakyrelu.o werror.o network.o neuron.o optim.o file.o fifo.o regularization.o

# ifeq ($(CC), gcc)
# 	CFLAGS = -W -Werror -Wall -Wextra -pedantic -Wno-pointer-arith -fPIC
# else ifeq ($(CC), clang)
# 	CFLAGS = -W -Werror -Wall -Wextra -pedantic -Wno-gnu-pointer-arith -fPIC
# endif

CFLAGS = -W -fPIC

ifeq ($(MODE), debug)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fno-omit-frame-pointer
else ifeq ($(MODE), debug-nvidia)
	DEBUG_FLAGS = -O0 -g -fsanitize=address -fsanitize-recover=address
else
	DEBUG_FLAGS = -O2
endif

# export CFLAGS
# export DEBUG_FLAGS

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
