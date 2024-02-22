CC	= gcc

SRC_DIR		:= ./

C_SRC	:= $(shell find $(SRC_DIR) -name '*.c' -print)

TARGET		:= libnet.so

CFLAGS	:= -Wall -O3 -g -fPIC
LDFLAGS	:= -shared

# DPDK
LIBDPDK_CFLAGS := $(shell pkg-config --cflags libdpdk)
LIBDPDK_LDFLAGS := $(shell pkg-config --libs libdpdk)
CXXFLAGS	+= $(LIBDPDK_CFLAGS)
LDFLAGS 	+= $(LIBDPDK_LDFLAGS)

all: $(TARGET)

C_STANDARD	:= -std=gnu11

OBJECTS 	:= $(C_SRC:.c=.o)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(C_STANDARD) $^ -o $@ $(LDFLAGS)

clean:
	@rm -rf $(OBJECTS) $(TARGET)
