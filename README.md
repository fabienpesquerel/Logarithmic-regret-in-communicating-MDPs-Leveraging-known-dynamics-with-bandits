# Logarithmic regret in communicating MDPs: Leveraging known dynamics with bandits: How to run the code

You can run our code by running `python3 main_SSH.py` from within the main directory called `average-reward-reinforcement-learning`.
This will fill the directory `./experiments/results` with plots and dump files that can be interpreted.

## Installing requirements

Depending on preferences, both a `requirements.txt` file and a `setup.py` file are given.

## Algorithms and Environments

Within the `main_SSH.py` you can decide the environment by uncommenting the one you want to test, decide the time horizon and the number of replicates.
You can also decide to comment/uncomment available algorithms.

## Windows system

If your system is Windows, please go to the `./experiments/utils.py` file, comment the line 14 and uncomment the line 13. Then you can proceed as usual.

## Parallelization

Be aware that this code will run on as much cores as there are available.
