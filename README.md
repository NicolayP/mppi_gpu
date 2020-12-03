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
  - [x] compute the expodential of the cost - beta only once.
  - [x] compute inverse of nabla and lambda only once.
  - [x] finish algo as mentioned in the point_mass.cu example.
  - [ ] adapt to run the code with different actions dimensions. 
  - [X] try mujoco simulation.
  - [ ] generalize the code.
  - [ ] start UUV integration (add for ros expert help).
  - [ ] write AUV math model.
  - [ ] start NN or GP integration.


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

  - seg fault due a bad useage of the cost pointer fixed.

  - completed algorithm. Need to send the data to the robot and shift the action vector by one unit and initalize U[T-1]. Haven't looked at the actual values of the controller. Implement get_inf to retrieve all the info, check the values and ultimately write a test code.

  - various corrections to generalize the code.
