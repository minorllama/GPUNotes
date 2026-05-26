# Variables
NVCC = /usr/bin/nvcc
NVCCFLAGS = -O3 -Xcompiler -fPIC -shared
LIB_NAME = libcppexts

# Target outputs
TARGET_RELEASE = $(LIB_NAME).so
TARGET_DEBUG = $(LIB_NAME)_dbg.so

# Default target builds both versions
all: $(TARGET_RELEASE) $(TARGET_DEBUG)

# Release build
$(TARGET_RELEASE): $(LIB_NAME).cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Debug build
$(TARGET_DEBUG): $(LIB_NAME).cu
	$(NVCC) $(NVCCFLAGS) -DDEBUG $< -o $@

# Clean up generated binaries
clean:
	rm -f $(TARGET_RELEASE) $(TARGET_DEBUG)

.PHONY: all clean