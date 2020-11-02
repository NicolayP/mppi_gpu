# MPPI GPU IMPLEMENTATION

## TODO:
  - [X] verify device execution.
  - [X] time comparaison with serial code.
  - [X] checking for memory leaks with cuda-memcheck.
    ```
      cuda-memcheck -leak-check full ./app
    ```
  - [X] add randomness to the threads.
    Currently looking at cuRAND.
    look at: Device cuRAND distribution functions.
  - [x] plot trajectories to see their validity.
  - [x] create cost class (get insp from the python class).
  - [x] add cost.
  - [x] implement generic reduction algorithm for sum and min.
  - [ ] finish algo as mentioned in the point_mass.cu example.
  - [ ] generalize the code.
  - [ ] try mujoco simulation.
  - [ ] start UUV integration (add for ros expert help).
  - [ ] write AUV math model.
  - [ ] start NN or GP integration.

## TODO:
    need to look at the returned value of the min and beta reduction function.
    This is probably the cause of the segfault at the end.

## Changes LOG:

  - At the current moment, the code computes 10E4 simulation of 200 steps within 300ms. The execution on my I9 cpu takes 530ms. There is more improvement to be made by sharing data between threads on the SM cache.

  - Installed cmake 18.4:
    ```
    cd ~/Downloads/cmake-3.18.4/   # or wherever you downloaded cmake
    ./bootstrap --prefix=$HOME/cmake-install
    make
    make install
    export PATH=$HOME/cmake-install/bin:$PATH
    export CMAKE_PREFIX_PATH=$HOME/cmake-install:$CMAKE_PREFIX_PATH
    ```
    This allowed to pass flags to the device linker. Linking with this allowed to use the features given by --arch sm_70.

  - Changed variables to class to pointer to class.
  called delet explicitly. Now no memory leak.

  - Added cuRAND lib to the project. This allows us to generate the input samples
  on the gpu. Removing the delay of transfering the data from the host to the device.
  - Need to pay attention to the rng_State. The state is copied on the local memory, then updated with the sampling process and copied back to the global memory after the computation.

  - added a reduction algorithm used for the min and add functions on a array. This is used to find the minimal cost and compute the normalisation term.
