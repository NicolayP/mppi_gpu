# Interface.

 controller input: current state (if memory correct) q_pos and q_vel, idealy in a single row vector q.

 controller output: Force/torque
   - Through thruster manager topic is /auv_command_output. type AUVCommand.
   - Without thruster manager topic is /thruster_output, type WrenchStamped.

 The controller is sampled based, idealy I don't want to write my own transition function thus do you think we can easily run in one headless simulation up to 3000 copies of the AUV with no collision and for 100~200 steps ? Each robot with different inputs.

 Idealy similar to [point mass gpu](https://github.com/NicolayP/mppi_gpu/blob/master/mppi_gpu/src/point_mass_gpu.cu). 
