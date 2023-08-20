archives = empty.o release.o fill.o null.o print.o

main: $(archives)

%.o: %.c
	$(CC) -c $(CFLAGS) $(DEBUG_FLAGS) $< -o ../../build/$@