# hycfd — general-purpose hypersonic CFD on MFEM (HDG compressible NS).
# Configuration discovery follows the other myapps makefiles.

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

# The Exasim run directory supplies the regression mesh/state and the
# generated model headers used as the physics oracle; the frozen replica app
# supplies the floored-model header and its Kokkos stub.
EXASIM_RUN_DIR ?= /home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline
FROZEN_APP ?= ../hdg_navierstokes
CPPFLAGS += -I. -DEXASIM_RUN_DIR=\"$(EXASIM_RUN_DIR)\" \
	-I$(EXASIM_RUN_DIR) -I$(FROZEN_APP)/m4_diagnostic
YAML_CXXFLAGS ?= $(shell pkg-config --cflags yaml-cpp 2>/dev/null)
YAML_LIBS ?= $(shell pkg-config --libs yaml-cpp 2>/dev/null)

CORE_OBJECTS := io/exasim_io.o io/exasim_mesh.o io/mesh_input.o \
	discretization/hdg_operator.o discretization/av_sensor.o \
	post/surface_post.o \
	physics/perfect_gas_model.o physics/physics_factory.o
BINARIES := hycfd test_physics test_mms test_ptc_limit test_parallel test_bc test_av

.PHONY: all check test test_g4 clean

all: $(BINARIES)

# Fast gates: physics units + HDG core + PTC limit.
check: test_physics test_mms test_ptc_limit test_bc test_av
	./test_physics
	./test_mms
	./test_ptc_limit
	./test_bc
	./test_av

# Full gates: fast gates + the frozen Mach 8 regression + the M4.5 sanity run.
MFEM_MPIEXEC ?= mpirun

test: check test_g4 hycfd input/cylinder_nr6_nc12_q4.mesh
	./hycfd -i input/mach8_regression.yaml --acceptance
	./hycfd -i input/m45_sanity.yaml --acceptance
	./hycfd -i input/m30_cylinder_file.yaml --acceptance

# G4 gates: np={2,4} consistency vs a serial reference, then the Mach 8
# regression at np=4.
test_g4: test_parallel hycfd
	./test_parallel --write parallel_reference.txt
	$(MFEM_MPIEXEC) -np 2 ./test_parallel --check parallel_reference.txt
	$(MFEM_MPIEXEC) -np 4 ./test_parallel --check parallel_reference.txt
	$(MFEM_MPIEXEC) -np 4 ./hycfd -i input/mach8_regression.yaml --acceptance

hycfd: driver/hycfd_main.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_physics: tests/test_physics.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_mms: tests/test_mms.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_ptc_limit: tests/test_ptc_limit.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_parallel: tests/test_parallel.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_bc: tests/test_bc.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

test_av: tests/test_av.o $(CORE_OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

tools/generate_cylinder_mesh: tools/generate_cylinder_mesh.o io/exasim_mesh.o io/exasim_io.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $^ -o $@ $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp

input/cylinder_nr6_nc12_q4.mesh: tools/generate_cylinder_mesh
	./tools/generate_cylinder_mesh 6 12 $@

%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(YAML_CXXFLAGS) $(CPPFLAGS) -c $< -o $@

io/exasim_io.o: io/exasim_io.cpp io/exasim_io.hpp
io/exasim_mesh.o: io/exasim_mesh.cpp io/exasim_mesh.hpp io/exasim_io.hpp
io/mesh_input.o: io/mesh_input.cpp io/mesh_input.hpp
discretization/hdg_operator.o: discretization/hdg_operator.cpp \
	discretization/hdg_operator.hpp physics/physics_model.hpp \
	io/exasim_io.hpp io/exasim_mesh.hpp
discretization/av_sensor.o: discretization/av_sensor.cpp \
	discretization/av_sensor.hpp discretization/hdg_operator.hpp \
	physics/physics_model.hpp
tests/test_av.o: tests/test_av.cpp discretization/av_sensor.hpp \
	discretization/hdg_operator.hpp physics/perfect_gas_model.hpp \
	io/exasim_mesh.hpp
physics/perfect_gas_model.o: physics/perfect_gas_model.cpp \
	physics/perfect_gas_model.hpp physics/physics_model.hpp \
	physics/perfect_gas.hpp
physics/physics_factory.o: physics/physics_factory.cpp \
	physics/physics_factory.hpp physics/perfect_gas_model.hpp \
	physics/physics_model.hpp physics/perfect_gas.hpp
post/surface_post.o: post/surface_post.cpp post/surface_post.hpp \
	discretization/hdg_operator.hpp physics/perfect_gas.hpp
driver/hycfd_main.o: driver/hycfd_main.cpp solvers/newton.hpp \
	solvers/trace_linear_solver.hpp discretization/hdg_operator.hpp \
	discretization/av_sensor.hpp \
	physics/perfect_gas.hpp io/exasim_mesh.hpp post/surface_post.hpp
tests/test_physics.o: tests/test_physics.cpp physics/perfect_gas.hpp \
	$(FROZEN_APP)/m4_diagnostic/my_model_floored.hpp
tests/test_mms.o: tests/test_mms.cpp solvers/newton.hpp \
	solvers/trace_linear_solver.hpp discretization/hdg_operator.hpp \
	physics/perfect_gas.hpp io/exasim_mesh.hpp
tests/test_ptc_limit.o: tests/test_ptc_limit.cpp solvers/newton.hpp \
	solvers/trace_linear_solver.hpp discretization/hdg_operator.hpp \
	physics/perfect_gas.hpp io/exasim_mesh.hpp
tests/test_parallel.o: tests/test_parallel.cpp solvers/newton.hpp \
	solvers/trace_linear_solver.hpp discretization/hdg_operator.hpp \
	physics/perfect_gas.hpp
tests/test_bc.o: tests/test_bc.cpp solvers/newton.hpp \
	solvers/trace_linear_solver.hpp discretization/hdg_operator.hpp \
	physics/perfect_gas_model.hpp physics/perfect_gas.hpp \
	io/exasim_mesh.hpp

clean:
	rm -f $(BINARIES) */*.o *~
