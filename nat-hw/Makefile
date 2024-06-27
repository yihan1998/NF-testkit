CC	= gcc

C_SRC	:= $(shell find ./ -name '*.c' -print)

TARGET		:= nat_hw

CFLAGS	:= -Wall -O3 -g -fno-stack-protector -fno-omit-frame-pointer -I/usr/include/ -I./
LDFLAGS	:= -L/usr/lib -lpthread

# Configurations
CONFIG 	= .config
include ${CONFIG}

ifdef CONFIG_NR_CPUS
	CFLAGS	+= -DCONFIG_NR_CPUS=$(CONFIG_NR_CPUS)
endif

# DPDK
LIBDPDK_CFLAGS := $(shell pkg-config --cflags libdpdk)
LIBDPDK_LDFLAGS := $(shell pkg-config --libs libdpdk)
CFLAGS	+= $(LIBDPDK_CFLAGS)
LDFLAGS	+= $(LIBDPDK_LDFLAGS)

# DOCA
LIBDOCA_CFLAGS := $(shell pkg-config --cflags doca)
LIBDOCA_LDFLAGS := $(shell pkg-config --libs doca)
CFLAGS	+= $(LIBDOCA_CFLAGS) -DDOCA_ALLOW_EXPERIMENTAL_API
LDFLAGS	+= $(LIBDOCA_LDFLAGS)

all: $(TARGET)

C_STANDARD	:= -std=gnu11

OBJECTS 	:= $(C_SRC:.c=.o)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(C_STANDARD) $^ -o $@ $(LDFLAGS)

clean:
	@rm -rf $(OBJECTS) $(TARGET)
