# MFEM HDG Navier--Stokes foundations - standalone Makefile.
# Configuration discovery follows myapps/convection_diffusion/makefile.

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

EXASIM_RUN_DIR ?= /home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline
CPPFLAGS += -DEXASIM_RUN_DIR=\"$(EXASIM_RUN_DIR)\" -I$(EXASIM_RUN_DIR)
YAML_CXXFLAGS ?= $(shell pkg-config --cflags yaml-cpp 2>/dev/null)
YAML_LIBS ?= $(shell pkg-config --libs yaml-cpp 2>/dev/null)

M1_TARGETS := test_physics mesh_convert_check
COMMON_OBJECTS := exasim_io.o exasim_mesh.o hdg_ns_operator.o

.PHONY: all test clean hdg_ns_driver test_hdg_mms

all: $(M1_TARGETS)

test: $(M1_TARGETS)
	./test_physics
	./mesh_convert_check $(EXASIM_RUN_DIR)

# These names are reserved by PLAN_HDG.md for M2. They deliberately do not
# build placeholders during M1.
hdg_ns_driver test_hdg_mms:
	$(error $@ belongs to Milestone M2; this checkout implements M1 only)

test_physics: test_physics.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

mesh_convert_check: mesh_convert_check.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $(CPPFLAGS) -c $< -o $@

test_physics.o: test_physics.cpp ns_physics.hpp
exasim_io.o: exasim_io.cpp exasim_io.hpp
exasim_mesh.o: exasim_mesh.cpp exasim_mesh.hpp exasim_io.hpp
hdg_ns_operator.o: hdg_ns_operator.cpp hdg_ns_operator.hpp exasim_io.hpp exasim_mesh.hpp
mesh_convert_check.o: mesh_convert_check.cpp exasim_io.hpp exasim_mesh.hpp hdg_ns_operator.hpp

clean:
	rm -f $(M1_TARGETS) *.o *~
