
ifeq ($(CC),)
CC=gcc
endif

clib:
	$(CC) truncation.c -fPIC -shared truncation.so