
.PHONY: all clean run

all:
	scons

clean:
	scons -c

run: all
	./aa
