# WEMA - Wavelet-based Eulerian Motion Amplification
# Makefile

CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L
LDFLAGS := -lm

# Directories
SRCDIR  := src
OBJDIR  := obj
BINDIR  := bin
TESTDIR := test

# Source files
SRCS    := $(wildcard $(SRCDIR)/*.c)
OBJS    := $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Target
TARGET  := $(BINDIR)/wema

# Default: release build
.PHONY: all
all: CFLAGS += -Ofast -DNDEBUG -march=native
all: $(TARGET)

# Debug build with Address Sanitizer
.PHONY: debug
debug: CFLAGS += -g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: $(TARGET)

# Debug without sanitizers (for valgrind)
.PHONY: debug-valgrind
debug-valgrind: CFLAGS += -g -O0
debug-valgrind: $(TARGET)

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Link executable
$(TARGET): $(OBJS) | $(BINDIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

# Include auto-generated dependencies
-include $(OBJS:.o=.d)

# Clean
.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install
.PHONY: install
install: all
	install -m 755 $(TARGET) /usr/local/bin/

# Help
.PHONY: help
help:
	@echo "WEMA - Wavelet-based Eulerian Motion Amplification"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Release build with optimizations (default)"
	@echo "  debug         - Debug build with AddressSanitizer"
	@echo "  debug-valgrind- Debug build without sanitizers (for valgrind)"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install to /usr/local/bin"
	@echo "  help          - Show this help"
