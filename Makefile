main:
	gcc -O2 -g -c src/wekua.c -o wekua.o

install:
	gcc -O2 -g -lOpenCL -shared wekua.o -o libwekua.so
	cp libwekua.so /usr/lib/

clean:
	rm -rf *.o