# --- Configuration ---
# Path to the 'rtec' compiler binary you built earlier
RTEC_BIN = ./easy-rte/rtec/rtec 

# Directory definitions
POLICY_DIR = policies
C_DIR = generated_c
LIB_DIR = lib

# Compiler flags for building shared libraries
CC = gcc
CFLAGS = -shared -fPIC -O3

# --- Automatic Discovery ---
# Find all .erte files (e.g., policies/OrderRateLimit.erte)
POLICIES = $(wildcard $(POLICY_DIR)/*.erte)

# Determine corresponding .so targets (e.g., lib/OrderRateLimit.so)
# We use the filename (without extension) to name the library
LIBRARIES = $(patsubst $(POLICY_DIR)/%.erte,$(LIB_DIR)/%.so,$(POLICIES))

# --- Targets ---

# Default target: Build all libraries
all: directories $(LIBRARIES)
	@echo "✅ All policies compiled successfully into $(LIB_DIR)/"

# Rule to create necessary directories
directories:
	@mkdir -p $(C_DIR)
	@mkdir -p $(LIB_DIR)

# Rule: How to build a .so file from an .erte file
# 1. Run rtec to generate .c and .h files in generated_c/
# 2. Run gcc to compile those .c files into a .so in lib/
$(LIB_DIR)/%.so: $(POLICY_DIR)/%.erte
	@echo "🔹 Compiling Policy: $<"
	
	# 1. Generate C code (Output naming convention matches easy-rte: F_PolicyName)
	$(RTEC_BIN) -l c -o $(C_DIR)/F_$* -i $<
	
	# 2. Compile to Shared Library
	$(CC) $(CFLAGS) -o $@ $(C_DIR)/F_$*.c
	
	@echo "   -> Created: $@"

# Clean up generated files
clean:
	rm -rf $(C_DIR) $(LIB_DIR)
	@echo "🗑️  Cleaned up generated files."

.PHONY: all clean directories