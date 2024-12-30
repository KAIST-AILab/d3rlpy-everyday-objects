# d3rlpy for Manipulating Everyday Objects
- This project is based on an open-source offline RL framework [d3rlpy](https://github.com/takuseno/d3rlpy). (MIT License)
- The objective of this project is to provide an integrated RL framework for manipulating every objects, such as deformable objects.

## Integrated Technique Lists
- This repository includes the outcomes of the following techniques produced by KAIST and Korea University, supported by the everyday object manipulation project (during Level 1, 2022~2024):
1. CORN (Cho et al., "CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects." ICLR 2024. [link](https://openreview.net/pdf/be6d29e6e7d18c8ea0250289f353011374d395b1.pdf))
2. DrilDICE (Seo et al., "Mitigating Covariate Shift in Behavioral Cloning via Robust Distribution Correction Estimation."  NeurIPS 2024. [link](https://openreview.net/pdf?id=lHcvjsQFQq))
3. PorelDICE (Kim et al., "Relaxed Stationary Distribution Correction Estimation for Improved Offline Policy Optimization" AAAI 2024. [link](https://ojs.aaai.org/index.php/AAAI/article/view/29218/30298))

## Instructions
Install this package by
```
git clone git@github.com:KAIST-AILab/d3rlpy-everyday-objects.git
cd d3rlpy-everyday-objects
pip install -e .
```

### Training SAC with CORN Representation
1. Clone `corn@iitp-d3rlpy` ([link](https://github.com/iMSquared/corn/tree/iitp-d3rlpy)) as in:
```
git clone --branch iitp-d3rlpy https://github.com/iMSquared/corn.git
```

2. Follow the docker build and follow the setup instructions in the [README](https://github.com/iMSquared/corn/tree/iitp-d3rlpy?tab=readme-ov-file).

3. Afterward, clone `yycho0108/d3rlpy-everyda-objects@imm-corn` repo and place it in an appropriate location within the docker image.

4. For integration with `d3rlpy`, refer to the provided sample scripts in `scripts/imm-corn`, where:

- `create_corn_dataset collects` the dataset
- `train_corn_sac_offline` uses the dataset to train SAC (*it's just a sample script, so the trained agent may not function well*)
- `train_corn_sac_online` trains SAC online on an environment with `num_env=1`. It would be very inefficient, so it's mostly meant for debugging.

While running the above scripts, you may need to configure data collection parameters, such as the number of parallel envs or the runtime device.

### Training DrilDICE
- Use the reproduction code contained in `reproductions/offline/drildice.py`.

### Training PorelDICE
- Use the reproduction code contained in `reproductions/offline/proel.py`.

## Acknowledgement
- This project  was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2022-II220311, Development of Goal-Oriented Reinforcement Learning Techniques for Contact-Rich Robotic Manipulation of Everyday Objects).
