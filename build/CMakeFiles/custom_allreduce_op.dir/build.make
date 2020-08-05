# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sandeep/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/sandeep/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build

# Include any dependencies generated for this target.
include CMakeFiles/custom_allreduce_op.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/custom_allreduce_op.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/custom_allreduce_op.dir/flags.make

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o: CMakeFiles/custom_allreduce_op.dir/flags.make
CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o: ../pyt_all_reduce_op.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o -c /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/pyt_all_reduce_op.cpp

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/pyt_all_reduce_op.cpp > CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.i

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/pyt_all_reduce_op.cpp -o CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.s

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o: CMakeFiles/custom_allreduce_op.dir/flags.make
CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o: ../pyt_all_reduce_kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/pyt_all_reduce_kernel.cu -o CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target custom_allreduce_op
custom_allreduce_op_OBJECTS = \
"CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o" \
"CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o"

# External object files for target custom_allreduce_op
custom_allreduce_op_EXTERNAL_OBJECTS =

libcustom_allreduce_op.so: CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o
libcustom_allreduce_op.so: CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o
libcustom_allreduce_op.so: CMakeFiles/custom_allreduce_op.dir/build.make
libcustom_allreduce_op.so: /home/sandeep/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so
libcustom_allreduce_op.so: /home/sandeep/anaconda3/lib/python3.7/site-packages/torch/lib/libc10.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/stubs/libcuda.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libnvrtc.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libnvToolsExt.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libcudart.so
libcustom_allreduce_op.so: /home/sandeep/anaconda3/lib/python3.7/site-packages/torch/lib/libc10_cuda.so
libcustom_allreduce_op.so: /home/sandeep/anaconda3/lib/python3.7/site-packages/torch/lib/libc10_cuda.so
libcustom_allreduce_op.so: /home/sandeep/anaconda3/lib/python3.7/site-packages/torch/lib/libc10.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libcufft.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libcurand.so
libcustom_allreduce_op.so: /usr/lib/x86_64-linux-gnu/libcublas.so
libcustom_allreduce_op.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libnvToolsExt.so
libcustom_allreduce_op.so: /usr/local/cuda/lib64/libcudart.so
libcustom_allreduce_op.so: CMakeFiles/custom_allreduce_op.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libcustom_allreduce_op.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/custom_allreduce_op.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/custom_allreduce_op.dir/build: libcustom_allreduce_op.so

.PHONY : CMakeFiles/custom_allreduce_op.dir/build

CMakeFiles/custom_allreduce_op.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/custom_allreduce_op.dir/cmake_clean.cmake
.PHONY : CMakeFiles/custom_allreduce_op.dir/clean

CMakeFiles/custom_allreduce_op.dir/depend:
	cd /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build /home/sandeep/misc/pytorch_custom_op/skballreduce-cpp-op/build/CMakeFiles/custom_allreduce_op.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/custom_allreduce_op.dir/depend
