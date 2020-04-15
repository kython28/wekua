main:
	gcc -fPIC -O2 -g -c src/wekua.c -o wekua.o
	gcc -O2 -g -c src/matrix.c -o matrix.o

install:
	gcc -fPIC -O2 -g -lOpenCL -shared wekua.o matrix.o -o libwekua.so
	cp libwekua.so /usr/lib/
	cp src/wekua.h /usr/include/wekua.h

clean:
	rm -rf *.o
