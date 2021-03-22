CC = gcc
CFLAGS = -W -g -fPIC -O2
archives = wekua.o matrix.o print.o trig.o blas.o extra.o aberth_root.o linear.o acti.o acti_linear.o acti_sigmoid.o acti_tanh.o acti_relu.o acti_leakyrelu.o werror.o wnetwork.o optim.o file.o

main: $(archives)
	$(CC) $(CFLAGS) -shared -lOpenCL $(archives) -o libwekua.so -lm

%.o: src/%.c
	$(CC) -c $(CFLAGS) $< -o $@

install:
	cp libwekua.so /usr/lib/
	cp src/wekua.h /usr/include/wekua.h
	rm -rf /usr/lib/wekua_kernels/
	cp -r kernels/ /usr/lib/wekua_kernels/
	chmod 755 /usr/lib/wekua_kernels
	chmod 644 /usr/lib/wekua_kernels/*
	chmod 644 /usr/include/wekua.h

clean:
	rm -rf $(archives)