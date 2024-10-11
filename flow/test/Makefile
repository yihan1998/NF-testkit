CC := gcc
LD := gcc

SRC_DIR		:= ./
BUILD_DIR	:= ./build
CFLAGS 		:= -Wall -Werror -O3 -g -march=native -I../common

LDFLAGS 	:= 

LIBDPDK_CFLAGS := $(shell pkg-config --cflags libdpdk)
LIBDPDK_LDFLAGS := $(shell pkg-config --libs libdpdk)
CFLAGS 		+= $(LIBDPDK_CFLAGS)
LDFLAGS 	+= $(LIBDPDK_LDFLAGS)

LIBDOCA_CFLAGS := $(shell pkg-config --cflags doca)
LIBDOCA_LDFLAGS := $(shell pkg-config --libs doca)
CFLAGS 		+= $(LIBDOCA_CFLAGS) -DDOCA_ALLOW_EXPERIMENTAL_API
LDFLAGS 	+= $(LIBDOCA_LDFLAGS)

SRCS		:= $(shell find ../common -name '*.c' -print) $(shell find ./ -name '*.c' -print)
OBJS 		= $(SRCS:%.c=$(BUILD_DIR)/%.o)
TARGET 		= $(BUILD_DIR)/test

C_STANDARD	:= -std=gnu11

MKDIR_P := mkdir -p

all: $(TARGET)

.PHONY: all

$(BUILD_DIR)/%.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(C_STANDARD) $(CFLAGS) -c $< -o $(@)

$(TARGET): $(OBJS)
	@echo ' '
	@echo 'Building $(@)'
	@echo 'Invoking Linker'
	@$(MKDIR_P) $(dir $@)
	$(LD) -o $(@) $^ $(LDFLAGS)
	@echo 'Finished building: $@'
	@echo ' '

clean:
	@rm -rf $(BUILD_DIR)
