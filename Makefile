COMPILER=gcc

main:
	$(COMPILER) -W -Wall -fPIC -O2 -march=native -c src/wekua.c -o wekua.o
	$(COMPILER) -W -Wall -fPIC -O2 -march=native -c src/matrix.c -o matrix.o
	$(COMPILER) -W -Wall -fPIC -O2 -march=native -c src/trig.c -o trig.o
	$(COMPILER) -W -Wall -fPIC -O2 -march=native -c src/roots.c -o roots.o

install:
	$(COMPILER) -W -Wall -fPIC -O2 -march=native -lOpenCL -shared wekua.o matrix.o trig.o roots.o -o libwekua.so -lm
	cp libwekua.so /usr/lib/
	cp src/wekua.h /usr/include/wekua.h

clean:
	rm -rf *.o
