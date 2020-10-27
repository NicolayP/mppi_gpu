# MPPI GPU IMPLEMENTATION

## TODO:
  - [X] verify device execution.
  - [X] time comparaison with serial code.
    - At the current moment, the code computes 10E4 simulation of 200 steps within 300ms. The execution on my I9 cpu takes 530ms. There is more improvement to be made by sharing data between threads on the SM cache.
  - [ ] plot trajectories to see their validity.
  - [ ] add randomness to the threads.
  - [ ] create cost class (get insp from the python class).
  - [ ] add cost.
  - [ ] finish algo as mentioned in the point_mass.cu example.
  - [ ] generalize the code.
  - [ ] try mujoco simulation.
  - [ ] start UUV integration (add for ros expert help).
  - [ ] write AUV math model.
  - [ ] start NN or GP integration.
