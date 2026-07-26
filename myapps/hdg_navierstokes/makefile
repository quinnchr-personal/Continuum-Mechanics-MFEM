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
CPPFLAGS += -DEXASIM_RUN_DIR=\"$(EXASIM_RUN_DIR)\" -I$(EXASIM_RUN_DIR) -Im4_diagnostic
YAML_CXXFLAGS ?= $(shell pkg-config --cflags yaml-cpp 2>/dev/null)
YAML_LIBS ?= $(shell pkg-config --libs yaml-cpp 2>/dev/null)

M1_TARGETS := test_physics mesh_convert_check
M2_TARGETS := hdg_ns_driver test_hdg_mms
M3_TARGETS := test_m3_reference
M4_TARGETS := test_ptc_limit
COMMON_OBJECTS := exasim_io.o exasim_mesh.o hdg_ns_operator.o

.PHONY: all test test_m1 test_m2 test_m3 test_m4 clean

all: $(M1_TARGETS) $(M2_TARGETS) $(M3_TARGETS) $(M4_TARGETS)

test: test_m1 test_m2 test_m3 test_m4

test_m1: $(M1_TARGETS)
	./test_physics
	./mesh_convert_check $(EXASIM_RUN_DIR)

test_m2: test_hdg_mms hdg_ns_driver
	./test_hdg_mms
	./hdg_ns_driver -i Input/input_m45_sanity.yaml --acceptance

test_m3: test_m3_reference hdg_ns_driver
	./test_m3_reference $(EXASIM_RUN_DIR)
	./hdg_ns_driver -i Input/input_mach8_baseline.yaml --acceptance

test_m4: test_ptc_limit hdg_ns_driver
	./test_ptc_limit
	./hdg_ns_driver -i Input/input_mach8_continuation.yaml --acceptance
	./hdg_ns_driver -i Input/input_mach8_continuation_analytic.yaml --acceptance
	./hdg_ns_driver -i Input/input_mach8_baseline.yaml --acceptance

test_physics: test_physics.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

mesh_convert_check: mesh_convert_check.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_hdg_mms: test_hdg_mms.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

hdg_ns_driver: hdg_ns_driver.o exasim_reference.o wall_post.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_m3_reference: test_m3_reference.o exasim_reference.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_ptc_limit: test_ptc_limit.o $(COMMON_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $(CPPFLAGS) -c $< -o $@

test_physics.o: test_physics.cpp ns_physics.hpp m4_diagnostic/my_model_floored.hpp
exasim_io.o: exasim_io.cpp exasim_io.hpp
exasim_mesh.o: exasim_mesh.cpp exasim_mesh.hpp exasim_io.hpp
hdg_ns_operator.o: hdg_ns_operator.cpp hdg_ns_operator.hpp ns_physics.hpp exasim_io.hpp exasim_mesh.hpp
hdg_ns_driver.o: hdg_ns_driver.cpp hdg_newton.hpp hdg_ns_operator.hpp ns_physics.hpp exasim_mesh.hpp wall_post.hpp
wall_post.o: wall_post.cpp wall_post.hpp hdg_ns_operator.hpp ns_physics.hpp
mesh_convert_check.o: mesh_convert_check.cpp exasim_io.hpp exasim_mesh.hpp hdg_ns_operator.hpp
test_hdg_mms.o: test_hdg_mms.cpp hdg_newton.hpp hdg_ns_operator.hpp ns_physics.hpp exasim_mesh.hpp
exasim_reference.o: exasim_reference.cpp exasim_reference.hpp exasim_mesh.hpp hdg_ns_operator.hpp ns_physics.hpp
test_m3_reference.o: test_m3_reference.cpp exasim_reference.hpp
test_ptc_limit.o: test_ptc_limit.cpp hdg_newton.hpp hdg_ns_operator.hpp ns_physics.hpp exasim_mesh.hpp
wall_post.o: wall_post.cpp wall_post.hpp hdg_ns_operator.hpp
hdg_ns_driver.o: hdg_ns_driver.cpp hdg_newton.hpp hdg_ns_operator.hpp ns_physics.hpp exasim_mesh.hpp exasim_reference.hpp wall_post.hpp

clean:
	rm -f $(M1_TARGETS) $(M2_TARGETS) $(M3_TARGETS) $(M4_TARGETS) *.o *~
