# Reinforcement Learning Model Training (with PPO)

The `./mujoco-setup/import-urdf.ipynb` Jupyter Notebook converts an Onshape-exported URDF definition of the balancing robot to a MuJoCo model with actuators, joints, and sensors.

`./cli.py` contains helper code to create, manage, train, evaluate and export the models.  
`./config.py` defines the curriculum learning stages.  
`./gym_env.py` defines the [Gymnasium environment](https://gymnasium.farama.org/api/env/), setting up the MuJoCo model.  
`./training.py` sets up the PPO algorithm from [SBX](https://github.com/araffin/sbx) with the training/evaluation environments.  
`./tuning.py` uses [Optuna](https://optuna.readthedocs.io/en/stable/) for hyperparameter optimization.  

## Usage
`./cli.py create test_model` Creates a new model  
`./cli.py train test_model -c 1_000_000` Trains the model for 1 million steps  
`./cli.py next test_model` 'Graduates' a model to the next version / curriculum phase  
`./cli.py eval test_model` Evaluates a trained model  
`./cli.py train_run test_model --to 12 -c 1_000_000` Trains the model, graduates, repeats, until it reaches version 12  
`./cli.py export test_model` Exports the model to ONNX, and quantizes+converts to .espdl format
`./cli.py infer test_model` Runs remote inference model, and collects real observation data in `./robot.log.new`
