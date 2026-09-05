# Continuum-Mechanics-MFEM flux-kernel framework.
# Builds lib$(LIBNAME).a from src/, then apps/ and tests/ against it.
# Configuration discovery follows the myapps makefiles (MFEM config.mk).

LIBNAME ?= cmf

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= $(MFEM_DIR)
MFEM_INSTALL_DIR ?= ../../mfem

CONFIG_MK ?= $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
	$(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk),\
	$(wildcard $(HOME)/MFEM/mfem/config/config.mk),\
	$(wildcard $(HOME)/mfem/config/config.mk),\
	$(wildcard /opt/mfem/config/config.mk),\
	$(wildcard /usr/local/mfem/config/config.mk),\
	$(shell for d in .. ../.. ../../.. ../../../.. ../../../../..; do \
		if [ -f $$d/config/config.mk ]; then echo $$d/config/config.mk; exit 0; \
		elif [ -f $$d/share/mfem/config.mk ]; then echo $$d/share/mfem/config.mk; exit 0; fi; \
	done))

ifneq ($(wildcard $(CONFIG_MK)),)
MFEM_CONFIG_DIR := $(dir $(CONFIG_MK))
MFEM_ROOT_FROM_CONFIG := $(patsubst %config/,%,$(MFEM_CONFIG_DIR))
MFEM_ROOT_FROM_CONFIG := $(patsubst %share/mfem/,%,$(MFEM_ROOT_FROM_CONFIG))
ifneq ($(filter command line environment,$(origin MFEM_BUILD_DIR)),command line environment)
MFEM_BUILD_DIR := $(MFEM_ROOT_FROM_CONFIG)
export MFEM_BUILD_DIR
endif
ifneq ($(filter command line environment,$(origin MFEM_DIR)),command line environment)
MFEM_DIR := $(MFEM_ROOT_FROM_CONFIG)
export MFEM_DIR
endif
-include $(CONFIG_MK)
else
$(warning MFEM config.mk not found. Set MFEM_BUILD_DIR or MFEM_INSTALL_DIR to locate it.)
endif

MFEM_CXX ?= g++
MFEM_FLAGS ?= -O2 -std=c++17
MFEM_LIBS ?=
MFEM_MPIEXEC ?= mpirun

YAML_CXXFLAGS ?= $(shell pkg-config --cflags yaml-cpp 2>/dev/null)
YAML_LIBS ?= $(shell pkg-config --libs yaml-cpp 2>/dev/null)

WARNFLAGS ?= -Wall
CPPFLAGS += -Isrc
CXXFLAGS_ALL := $(MFEM_FLAGS) $(YAML_CXXFLAGS) $(WARNFLAGS) $(CPPFLAGS)
LINK_LIBS := $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

LIB := lib$(LIBNAME).a
LIB_SRC := $(sort $(wildcard src/*/*.cpp))
LIB_OBJ := $(LIB_SRC:.cpp=.o)
APP_SRC := $(sort $(wildcard apps/*.cpp))
APPS := $(APP_SRC:.cpp=)
TEST_SRC := $(sort $(wildcard tests/*.cpp))
TESTS := $(TEST_SRC:.cpp=)
DEPS := $(LIB_OBJ:.o=.d) $(APP_SRC:.cpp=.d) $(TEST_SRC:.cpp=.d)

.PHONY: all lib apps tests check test clean

all: lib apps tests

lib: $(LIB)
apps: $(APPS)
tests: $(TESTS)

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^

$(APPS): apps/%: apps/%.o $(LIB)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(LIB) $(LINK_LIBS)

$(TESTS): tests/%: tests/%.o $(LIB)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(LIB) $(LINK_LIBS)

%.o: %.cpp
	$(MFEM_CXX) $(CXXFLAGS_ALL) -MMD -MP -c $< -o $@

-include $(DEPS)

# Fast gates (S1-S3): serial unit and MMS tests.
CHECK_TESTS := tests/test_base tests/test_materials tests/test_solid_mms
check: $(CHECK_TESTS)
	@for t in $(CHECK_TESTS); do echo "== $$t"; ./$$t || exit 1; done

# Full gates (S4): fast gates, the YAML-driven app runs serial and np=4,
# np={2,4} consistency vs a serial reference, and the benchmarks with the
# frozen Cook's membrane regression value, serial and np=4.
test: check apps/solid_mechanics tests/test_benchmarks tests/test_parallel
	./apps/solid_mechanics -i apps/input/cook.yaml
	$(MFEM_MPIEXEC) -np 4 ./apps/solid_mechanics -i apps/input/cook.yaml
	./apps/solid_mechanics -i apps/input/cantilever3d.yaml
	mkdir -p tests/out
	./tests/test_parallel --write tests/out/parallel_reference.txt
	$(MFEM_MPIEXEC) -np 2 ./tests/test_parallel --check tests/out/parallel_reference.txt
	$(MFEM_MPIEXEC) -np 4 ./tests/test_parallel --check tests/out/parallel_reference.txt
	./tests/test_benchmarks
	$(MFEM_MPIEXEC) -np 4 ./tests/test_benchmarks
	./tests/test_benchmarks --cook-ratio-gate

clean:
	rm -f $(LIB) $(LIB_OBJ) $(APPS) $(TESTS) $(DEPS) apps/*.o tests/*.o
