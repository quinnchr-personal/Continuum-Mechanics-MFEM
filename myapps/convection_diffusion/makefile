# Convection-diffusion driver - standalone Makefile
# Attempts to locate MFEM's config.mk by environment or by searching parent dirs.

MFEM_DIR ?= ../..
# Allow MFEM to be set via MFEM_BUILD_DIR or MFEM_DIR; keep a sensible default
MFEM_BUILD_DIR ?= $(MFEM_DIR)
MFEM_INSTALL_DIR ?= ../../mfem

# Look for config/config.mk in common build/install locations and parent directories
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

# If we found a config.mk path, derive MFEM root and set MFEM vars before including it
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
# Fallback defaults so `make` gives a clearer error when MFEM isn't configured.
# These are safe defaults and will be overridden if `config.mk` is included.
MFEM_CXX ?= g++
MFEM_FLAGS ?= -O2 -std=c++17
MFEM_LIBS ?=

YAML_CXXFLAGS ?= $(shell pkg-config --cflags yaml-cpp 2>/dev/null)
YAML_LIBS ?= $(shell pkg-config --libs yaml-cpp 2>/dev/null)

TARGETS = linear_convection_diffusion_1D linear_convection_diffusion_2D linear_convection_diffusion_2D_circle nonlinear_convection_diffusion_1D ablation_test_case1_2D ablation_test_case2_1_2D ablation_test_case2_2_2D ablation_test_case2_2_2D_test ablation_qstar_blowing_1D ale_validation_be ale_validation_be_stability ale_validation_be_convergence ale_validation_be_accuracy

.PHONY: all clean

all: $(TARGETS)

linear_convection_diffusion_1D: linear_convection_diffusion_1D.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $< -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

linear_convection_diffusion_2D: linear_convection_diffusion_2D.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $< -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

linear_convection_diffusion_2D_circle: linear_convection_diffusion_2D_circle.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $< -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

nonlinear_convection_diffusion_1D: nonlinear_convection_diffusion_1D.cpp newton_petsc_solver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $< -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

ablation_test_case1_2D: ablation_test_case1_2D.cpp tacot_material.cpp tacot_material.hpp newton_petsc_solver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) ablation_test_case1_2D.cpp tacot_material.cpp -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

ablation_test_case2_1_2D: ablation_test_case2_1_2D.cpp tacot_material.cpp tacot_material.hpp bprime_table.cpp bprime_table.hpp surface_bc_schedule.cpp surface_bc_schedule.hpp newton_petsc_solver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) ablation_test_case2_1_2D.cpp tacot_material.cpp bprime_table.cpp surface_bc_schedule.cpp -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

ablation_test_case2_2_2D: ablation_test_case2_2_2D.cpp tacot_material.cpp tacot_material.hpp bprime_table.cpp bprime_table.hpp surface_bc_schedule.cpp surface_bc_schedule.hpp mesh_recession_handler.cpp mesh_recession_handler.hpp newton_petsc_solver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) ablation_test_case2_2_2D.cpp tacot_material.cpp bprime_table.cpp surface_bc_schedule.cpp mesh_recession_handler.cpp -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

ablation_test_case2_2_2D_test: ablation_test_case2_2_2D_test.cpp tacot_material.cpp tacot_material.hpp bprime_table.cpp bprime_table.hpp surface_bc_schedule.cpp surface_bc_schedule.hpp mesh_recession_handler.cpp mesh_recession_handler.hpp newton_petsc_solver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) ablation_test_case2_2_2D_test.cpp tacot_material.cpp bprime_table.cpp surface_bc_schedule.cpp mesh_recession_handler.cpp -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

ablation_qstar_blowing_1D: ablation_qstar_blowing_1D.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

ale_validation_be: ale_validation_be.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

ale_validation_be_stability: ale_validation_be_stability.cpp ale_validation_be.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) ale_validation_be_stability.cpp -o $@ $(MFEM_LIBS)

ale_validation_be_convergence: ale_validation_be_convergence.cpp ale_validation_be.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) ale_validation_be_convergence.cpp -o $@ $(MFEM_LIBS)

ale_validation_be_accuracy: ale_validation_be_accuracy.cpp ale_validation_be.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) ale_validation_be_accuracy.cpp -o $@ $(MFEM_LIBS)

clean:
	rm -f $(TARGETS) *.o *~
