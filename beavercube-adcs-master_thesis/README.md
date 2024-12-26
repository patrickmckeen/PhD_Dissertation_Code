# beavercube-adcs
ADCS repo for BeaverCube, mostly for the simulation

Update 2/16/21:

Any old Simulink files or MATLAB code can be found in the "master-archived" branch of this repository.

The simulation is now in bcadcs.slx. It has many parameters and constants, which you must configure by running call_simulation.m. This script should initialize all variables (including loading from m-files), run the simulation, and then generate some plots once the simulation is completed.

The "figures" folder has figures, the "matfiles" folder has m-files with useful data, and the "supplemental-scripts" folder has useful MATLAB scripts pertaining to the trajectory optimizer (a MATLAB version of the trajectory optimizer + some useful debugging scripts), and some additional reference scripts.

The actual trajectory optimization code is in the "clean-rpi" branch of this repository, with the working branch as "clean-rpi-dev". In order to run the simulation with the trajectory optimizer, you will need to have a local version of the "clean-rpi" branch on your computer. I suggest making a "starlab-beavercube-adcs" folder containing a "beavercube-adcs" folder (this branch) and a "clean-rpi" folder (that branch).

You should not encounter file path errors in the simulation, but if you face issues calling the trajectory optimizer from MATLAB, I recommend modifying all the path names in the "trajOpt" method in "altro_general.cpp" in the trajectory optimizer to point directly to the location of each file on your computer, rather than using relative pathnames. I have encountered this bug before only when calling the trajectory optimizer from MATLAB. You should also test that you can compile and run the trajectory optimizer code from the command line before attempting to run from MATLAB.

You may also need to modify "controller.m" inside the "controller" block of "bcadcs.slx", although relative pathnames should work fine there.

____________________________________________________________________________________________________________________


Update 9/23/20:

The simulation is now in controller_testbed.slx. It has many parameters and constants, which you must configure by running simulation_initialize.m. This process involves loading some data from the "matfiles" folder, just FYI.

Patrick's ALTRO implementation is in ALILQR_BC.m, in the main project folder.

If you are looking for graphs, they can be found in the "figures" folder.

If you are looking for C++ code, look in TrajectoryPlanning -> root. The TrajectoryPlanning/root/matrix/include contains the PX4 header-only matrix library, slightly modified by Alex to be more useful for us. TrajectoryPlannning/root/MatrixTest is where the actual code is. In this folder:

    -test.cpp is the "main file", and it runs tests on various ALTRO functions from other files
    -altro_general.cpp is where the "main part" of ALTRO lives, and is in progress. it is imported by test.cpp
    -altro_beavercube.cpp is where the "beavercube-specific" part of ALTRO lives (like dynamics functions, etc). it is imported by altro_general.cpp
    -altro_helper.cpp is where basically just ALTRO helper functions are, primarily so the other files are easier to read. it is imported by altro_beavercube.cpp
    
To test the C++ code, aka get it to run, navigate to the "root" directory on Terminal and run "cmake .", then run "make", then cd into the MatrixTest folder, then run ".\MatrixTest".

If you are looking for anything else, look in old_items or old scripts 9-11-20.
____________________________________________________________________________________________________________________

Update 5/28:

The simulation is in test_simulation.slx. Its dependencies are dynamics.slx, sensor.m, and software.m. It also requires preloading some data from v_ECI.mat, r_ECI.mat, sun_ECI.mat, and magfield_ECI.mat, and generating some random initial conditions from ekf_test_initialconditions.m.

If you are looking for something else, the separated ekf, controller, and command scripts for testing are in the "scripts" folder, and all previous work, including old Simulink modules, old Simulink testbeds, and detumbling simulation scripts, is in the "old-items" folder.

To run the ADCS simulation, follow these steps:

1) Import v_ECI.mat, r_ECI.mat, sun_ECI.mat, and magfield_ECI.mat into your Matlab workspace. You can do this by right clicking them and selecting "Open with...Matlab". This will allow the simulation to access the orbit propagation data from STK.

2) Open ekf_test_initialconditions.m with Matlab and run it. This will generate a random initial attitude q. It will also generate an initial rotation rate w, which is equal to the orbital rotation rate + a random rotation rate with magnitude ~0.5 deg/s. These values represent BeaverCube's state after detumbling. It will also generate an initial set of gyroscope biases and initialize the EKF's guess of BeaverCube's state.

3) Open test_simulation.slx and run it. You may need to manually import dynamics.slx if Simulink can't find it. You can do this by clicking on the dynamics block, selecting "Open..." and navigating to dynamics.slx in the beavercube-adcs folder.

It should now run successfully!
