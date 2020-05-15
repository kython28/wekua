main:
	gcc -W -Wall -fPIC -O2 -march=native -g -c src/wekua.c -o wekua.o
	gcc -W -Wall -O2 -march=native -g -c src/matrix.c -o matrix.o

install:
	gcc -W -Wall -fPIC -O2 -march=native -g -lm -lOpenCL -shared wekua.o matrix.o -o libwekua.so
	cp libwekua.so /usr/lib/
	cp src/wekua.h /usr/include/wekua.h

clean:
	rm -rf *.o
