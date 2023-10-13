CC = gcc
CFLAGS = -W -g -fPIC -O2
archives = wekua.o matrix.o print.o trig.o blas.o extra.o aberth_root.o linear.o acti.o acti_linear.o acti_sigmoid.o acti_tanh.o acti_softmax.o acti_relu.o acti_leakyrelu.o werror.o network.o neuron.o optim.o file.o fifo.o regularization.o

main: $(archives)
	$(CC) $(CFLAGS) -shared -lOpenCL -pthread $(archives) -o libwekua.so -lm

%.o: src/%.c
	$(CC) -c $(CFLAGS) $< -o $@

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
	rm -rf $(archives)