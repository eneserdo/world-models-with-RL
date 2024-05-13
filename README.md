# RL based World Models

A fork to train the world models with various RL algorithms and compare their performance on different gym environments.

## Installation

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge swig
pip install gymnasium[all]
pip install -r requirements.txt
```

## TODO

- [ ] Train in dream environment: Currently it just uses the VAE and MDRNN for feature extraction and context vector calculation. Create modified version of RolloutGenerator class and train it with CMA-ES. 
- [ ] Train the model in alternative environments: Change the model architecture. 
- [ ] Train the model with RL algorithms: Wrap the RolloutGenerator class to create a custom environment compatible with stable-baselines3. Check the link for more details: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

- [ ] Train the model iteratively: Use the trained model's policy to generate data and train the MDN-RNN with the new generated data.


**NOTE**: Please check the README.md of the forked repo.