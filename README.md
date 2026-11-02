# rl-robot
Training setup & firmware for ESP32-S3 based balancing robot, using PPO reinforcement learning.

## training
The `./training` subdirectory contains the Python code for training, evaluating, exporting the neural networks. It uses [SBX](https://github.com/araffin/sbx) to performantly train the PPO model with Jax, using a simplified version of the balancing robot simulated in MuJoCo. Exported models are quantized with [esp-ppq](https://github.com/espressif/esp-ppq) to run on the ESP32-S3.

## firmware
Platformio project to build firmware for the ESP32-S3, using [esp-dl](https://docs.espressif.com/projects/esp-dl/en/latest/introduction/readme.html) for model inference, and FreeRTOS tasks for collecting input data (see `./firmware/src/inputs.hpp).

## License
GPL-3.0

`./firmware/vendor` contains KalmanFilter.hpp adapted from code by Korneliusz Jarzebski (GPL-3.0), and [esp-dl](https://github.com/espressif/esp-dl) by Espressif Systems Co., Ltd. (MIT).

The initial project was loosely based on the [tutorial series](https://www.digikey.com/en/maker/tutorials/2026/reinforcement-learning-for-robotics-part-1-cad-to-mujoco-simulator) and [code](https://github.com/ShawnHymel/reinforcement-learning-for-robotics) by Shawn Hymel (Apache-2.0).
