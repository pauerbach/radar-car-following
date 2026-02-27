# Sim-to-Real Transfer of an End-to-End Reinforcement Learning Policy for Radar-Based Car-Following

This repository contains the source code accompanying the paper  
**“Sim-to-Real Transfer of an End-to-End Reinforcement Learning Policy for Radar-Based Car-Following”**,  
submitted to the IEEE International Conference on Intelligent Transportation Systems (ITSC) 2026.

## About the work

This paper investigates the sim-to-real transfer of an end-to-end reinforcement learning policy for radar-based car-following. The proposed approach learns longitudinal control directly from range–Doppler representations produced by an automotive radar, avoiding hand-crafted perception features or explicit target tracking. A physics-inspired radar simulation is developed to generate realistic range–Doppler maps under varying relative motion between a leading and a following vehicle. Using this simulated sensor model, a Proximal Policy Optimization (PPO) agent is trained to regulate vehicle speed in order to maintain a desired time-to-collision with a stochastic lead vehicle.
To evaluate real-world applicability, the trained policy is deployed without retraining on a model-scale vehicle equipped with an Infineon BGT60TR13C radar. The sim-to-real transition is facilitated by aligning the simulated and real radar processing pipelines and by incorporating representative noise and resolution characteristics during training. Experimental results demonstrate that the policy generalizes to real radar measurements and achieves stable car-following behavior under varying lead vehicle speeds.

[Videos of agent evaluation](https://www.youtube.com/playlist?list=PL3U0Q5QF11vZQqoIPdr3Y9lDzQej6ik_s)

---

## Installation

This repository uses the [Pixi](https://pixi.prefix.dev/latest/) dependency manager.

First, install Pixi by following the official [installation guide](https://pixi.prefix.dev/latest/installation/).

Two Pixi environments are provided:

- **`cpu`**  
  Installs CPU-only dependencies along with the required ROS2 packages.  
  Intended for:
  - Policy evaluation in simulation  
  - Real-world evaluation using ROS2  

- **`gpu`**  
  Installs PyTorch with CUDA support.  
  Intended for:
  - Training the reinforcement learning agent on NVIDIA GPU-equipped systems  

Activate an environment using on of the following commands:

```bash
pixi shell -e cpu  # Activate CPU environment
pixi shell -e gpu  # Activate GPU environment
```

## Training an agent

To train an agent, activate the GPU environment:

```
pixi shell -e gpu
```

The main training script is:

```
train_ppo.py
```

All training hyperparameters and environment settings are defined in:

```
config.yaml
```

Modify this file before starting training to adjust the desired training configuration.

Training progress is logged using wandb (Weights & Biases).
To enable logging:

1. Create a Weights & Biases account.

2. Configure wandb locally (e.g., via wandb login).

## Evaluating an Agent in Simulation

To evaluate a trained agent in simulation, use:

```
enjoy_ppo.py
```

Before running the script:

- Specify the path to the desired model checkpoint.

- Adjust the environment configuration as needed.

During evaluation, two NumPy files are generated:

- ttc.npy – Time-to-Collision values

- headway.npy – Time headway values

These files can be visualized using the plotting scripts in the `results/` directory.

## Evaluating an agent in the real-world testbed

To evaluate a trained agent on the miniature vehicle platform, use:

```
ros/run_ros2_real_radar.py
```

#### Step 1: Source and Build the ROS2 Environment

```
cd ros
./build.sh
source ros/install/setup.zsh
```

#### Step 2: Run the Evaluation Script

The script allows you to configure whether a real or simulated radar sensor is used.

**Recording Data**

To record all ROS2 messages during evaluation:

```
ros2 bag record --all
```

This creates a ROS2 bag file containing all recorded data.

**Post-Processing Recorded Data**

To evaluate a recorded ROS2 bag:

Start the evaluation script:

```
ros/evaluate_rosbag.py
```

In a separate terminal, play back the bag file:

```
ros2 bag play <ROS2BagFolder> --loop
```

After terminating the evaluation script, the same NumPy files (ttc.npy and headway.npy) are generated as in the simulation evaluation. These can be visualized using the same plotting tools.

## Plotting Results

The script:

```
results/plot_ttc.py
```

is used to visualize and compare TTC and headway distributions for different follower types.
It reproduces the plots presented in the paper.
