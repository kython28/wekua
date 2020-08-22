COMPILER=gcc
ARGS=-W -Wall -fPIC -O2 -march=native

main:
	$(COMPILER) $(ARGS) -c src/wekua.c -o wekua.o
	$(COMPILER) $(ARGS) -c src/matrix.c -o matrix.o
	$(COMPILER) $(ARGS) -c src/trig.c -o trig.o
	$(COMPILER) $(ARGS) -c src/roots.c -o roots.o
	$(COMPILER) $(ARGS) -c src/activation.c -o activation.o
	$(COMPILER) $(ARGS) -c src/linear.c -o linear.o
	$(COMPILER) $(ARGS) -c src/sequential.c -o sequential.o
	$(COMPILER) $(ARGS) -c src/arch.c -o arch.o
	$(COMPILER) $(ARGS) -c src/loss.c -o loss.o
	$(COMPILER) $(ARGS) -c src/optim.c -o optim.o

install:
	$(COMPILER) $(ARGS) -lOpenCL -shared wekua.o matrix.o trig.o roots.o activation.o linear.o sequential.o arch.o loss.o optim.o -o libwekua.so -lm
	cp libwekua.so /usr/lib/
	cp src/wekua.h /usr/include/wekua.h
	rm -rf /usr/lib/wekua_kernels/
	cp -r kernels/ /usr/lib/wekua_kernels/
	chmod 755 /usr/lib/wekua_kernels
	chmod 644 /usr/lib/wekua_kernels/*

clean:
	rm -rf *.o
