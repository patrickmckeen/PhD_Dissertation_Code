# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.25.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.25.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build

# Include any dependencies generated for this target.
include CMakeFiles/pysat.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pysat.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pysat.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pysat.dir/flags.make

CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o: CMakeFiles/pysat.dir/flags.make
CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o: /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/planner/Satellite.cpp
CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o: CMakeFiles/pysat.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o -MF CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o.d -o CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o -c /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/planner/Satellite.cpp

CMakeFiles/pysat.dir/src/planner/Satellite.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pysat.dir/src/planner/Satellite.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/planner/Satellite.cpp > CMakeFiles/pysat.dir/src/planner/Satellite.cpp.i

CMakeFiles/pysat.dir/src/planner/Satellite.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pysat.dir/src/planner/Satellite.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/planner/Satellite.cpp -o CMakeFiles/pysat.dir/src/planner/Satellite.cpp.s

# Object files for target pysat
pysat_OBJECTS = \
"CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o"

# External object files for target pysat
pysat_EXTERNAL_OBJECTS =

pysat.cpython-310-darwin.so: CMakeFiles/pysat.dir/src/planner/Satellite.cpp.o
pysat.cpython-310-darwin.so: CMakeFiles/pysat.dir/build.make
pysat.cpython-310-darwin.so: /usr/local/lib/libarmadillo.dylib
pysat.cpython-310-darwin.so: /usr/local/lib/libarmadillo.dylib
pysat.cpython-310-darwin.so: libsat.a
pysat.cpython-310-darwin.so: /usr/local/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib
pysat.cpython-310-darwin.so: /usr/local/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib
pysat.cpython-310-darwin.so: /usr/local/lib/libarmadillo.dylib
pysat.cpython-310-darwin.so: /usr/local/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib
pysat.cpython-310-darwin.so: CMakeFiles/pysat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module pysat.cpython-310-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pysat.dir/link.txt --verbose=$(VERBOSE)
	/Library/Developer/CommandLineTools/usr/bin/strip -x /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build/pysat.cpython-310-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/pysat.dir/build: pysat.cpython-310-darwin.so
.PHONY : CMakeFiles/pysat.dir/build

CMakeFiles/pysat.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pysat.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pysat.dir/clean

CMakeFiles/pysat.dir/depend:
	cd /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build /Users/patrickmckeen/Documents/GitHub/McKeenADS/GeneralizedADS/ADCS/trajectory_planner/src/build/CMakeFiles/pysat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pysat.dir/depend
