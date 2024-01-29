# Introduction
This repo provides source code for the paper "MalLight: Coordinated Traffic Signal Control for Traffic Signal Malfunctions", which pioneers the application of a Reinforcement Learning(RL)-based approach, called MalLight, to address the challenges posed by traffic signal malfunction.

Our implementation is based on a framework called [LibSignal](https://darl-libsignal.github.io/). LibSignal is a more comprehensive toolkit for developing simulation-based traffic signal control model.

# Install
## Python dependencies
Our code is based on Python 3.9 and Pytorch 1.11.0. Please install dependencies by running:
```
pip install -r requirements.txt
```
## Simulator environment
The "Simulation of Urban MObility" (SUMO) environment is required to run experiments. Please refer to [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#) for installment.

# Run 
You can start experiments of MalLight on NY7x7 dataset by running:
```
python run.py --agent mallight
```

# Citation
Paper is under review.