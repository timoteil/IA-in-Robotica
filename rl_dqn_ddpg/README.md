# From Q-tables to DDPG

A bridge between two other folders in this directory.

* `../rl_course_1h/` ends with a tabular Q-learner on a 24-cell gridworld and a small neural Q in numpy.
* `../SO101_PickCube/` trains a robot arm with PPO, through a library.

Between them there is a gap: the state becomes a vector of real numbers, the action may become a real number, and the learner becomes a pair of neural networks. This module covers the gap in four labs. Every reinforcement-learning component is written in the lab files, in plain Python: the replay buffer, the TD target, the target networks, the exploration noise, the actor and critic updates, and the training loop. PyTorch is used for the networks and for autograd only. There is no Stable-Baselines, no RLlib, no CleanRL, and no other RL library.

```
rl_bridge_dqn_ddpg/
├── requirements.txt
└── labs/
    ├── common.py                          replay buffer, mlp, soft_update, evaluate, CSV log, plot helpers
    ├── lab1_tabular_mountaincar.py        the table of the 1-hour course, on a state of two real numbers
    ├── lab2_dqn_mountaincar.py            the table becomes a network: DQN
    ├── lab3_ddpg_inverted_pendulum.py     the action becomes a real number: DDPG on a MuJoCo pendulum
    ├── lab4_ddpg_mountaincar_continuous.py  the same DDPG fails on a hard-exploration task, and the cure
    ├── play.py                            watch a trained controller (window or mp4)
    └── outputs/                           figures, CSV learning curves, checkpoints, videos
```

Each lab changes one thing from the one before it.

| Lab | Task | What changes | What it teaches |
|---|---|---|---|
| 1 | MountainCar-v0, 3 actions | the state is two real numbers; boxes turn it into table rows | the table still works, and grows as N^d |
| 2 | MountainCar-v0, 3 actions | the table becomes a network | replay buffer, target network, epsilon-greedy |
| 3 | InvertedPendulum-v5 (MuJoCo), 1 real action | `argmax_a Q` can no longer be enumerated | actor + critic, target networks, noise: DDPG |
| 4 | MountainCarContinuous-v0 | only the environment: rewards are sparse | exploration: why the same agent learns to stand still |

The labs share the style of `rl_course_1h/labs`: a long docstring at the top of each file (the idea, the update equation, "Run it", "Exercises"), flags for the experiments, a printed trace of one update that can be checked by hand, a PNG in `labs/outputs/`, and `--show` to also open the plot.

## Setup

Python 3.12 or 3.13. PyTorch has to support your Python version.

macOS / Linux:

```bash
cd rl_bridge_dqn_ddpg
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cd labs
```

Windows (PowerShell), same as in `SO101_PickCube`:

```powershell
cd rl_bridge_dqn_ddpg
uv venv --python 3.13 .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
cd labs
```

If you already built the `SO101_PickCube/.venv`, it has everything needed except `pygame` (used only by `play.py --lab 2` and `--lab 4` to draw the car): `uv pip install pygame`.

Everything runs on a CPU, no GPU needed. Runs are seeded and use one thread, so the numbers below can be reproduced on the same machine and library versions; other machines give the same behaviour, with different decimals.

## Lab 1: tabular Q-learning with boxes

```bash
python lab1_tabular_mountaincar.py                 # 20 x 20 boxes, about 20 seconds
python lab1_tabular_mountaincar.py --trace 8       # print the first 8 updates
python lab1_tabular_mountaincar.py --bins 8        # coarse;  --bins 40: fine
python lab1_tabular_mountaincar.py --epsilon 0     # no random exploration
python lab1_tabular_mountaincar.py --epsilon 0 --q0 -150    # ... and a pessimistic table
```

Expected (seed 0): the table has 1,200 entries. The flag is first reached in episode 3291 (seeds 1 and 2: 3329 and 3055). The greedy policy reaches the flag from all 20 starts with a mean return of -157 (seeds 1 and 2: -136 and -161). With `--bins 8` the greedy policy reaches the flag in 0% of runs, with `--bins 40` in 25%. With `--epsilon 0` the agent still finds the flag, earlier (episode 498), because a table of zeros under a reward of -1 makes every tried action look worse than the untried ones. With `--q0 -150` it never does. The figure shows the learning curve, the value map (a spiral in position-velocity space) and the greedy action map.

## Lab 2: DQN

```bash
python lab2_dqn_mountaincar.py                     # about 30 seconds
python lab2_dqn_mountaincar.py --trace             # one minibatch update, row by row
python lab2_dqn_mountaincar.py --no-replay         # train on the newest transition only
python lab2_dqn_mountaincar.py --no-target         # compute y from the network being trained
python lab2_dqn_mountaincar.py --compare           # all three on one plot, about 80 seconds
python lab2_dqn_mountaincar.py --no-random-starts  # always start at the bottom of the hill
```

Expected (seed 0): the greedy policy reaches the flag from all 20 starts with a mean return of -113, better than the table of Lab 1 (-157) with 17,283 weights instead of 1,200 entries. On seeds 1, 2 and 3 the last 5-episode evaluation is -130, -153 and -97, and the best one is about -97 to -106; the curve is noisy from one evaluation to the next. In `--compare`, `--no-replay` stays at -200 throughout: it never learns. `--no-target` does learn here, and its evaluation curve is visibly more erratic than the default's; on this small problem the target network matters less than the replay buffer, which is itself a result worth discussing. With `--no-random-starts` the agent never sees a single flag transition in 100,000 steps and the result is -200 (0% of runs reach the flag): see exercise 7 in the file.

Training episodes begin at random positions and velocities on the hill. This is stated in the docstring and is the one concession to the environment: a network that starts every episode at the bottom and pushes at random never reaches the flag. Evaluation always starts at the bottom.

## Lab 3: DDPG on the inverted pendulum

```bash
python lab3_ddpg_inverted_pendulum.py              # about 40 seconds
python lab3_ddpg_inverted_pendulum.py --trace      # one update, with the gradient dQ/da
python lab3_ddpg_inverted_pendulum.py --td3        # the three TD3 repairs
python lab3_ddpg_inverted_pendulum.py --twin       # ... or any one of --twin --delay --smooth
python lab3_ddpg_inverted_pendulum.py --noise ou   # correlated exploration noise
python lab3_ddpg_inverted_pendulum.py --seeds 5    # how many of 5 seeds end at return 1000
python play.py --lab 3                             # write outputs/play_lab3.mp4
```

Expected (seed 0): the greedy evaluation first reaches 1000 (the maximum, the pole never falls) at step 14,000 and stays there; the final 20-episode evaluation is 1000 for all 20. Training takes about 40 seconds for 30,000 steps. With `--seeds 5`, **4 of 5 seeds** end at 1000 (seed 4 does not learn in 30,000 steps); with `--td3 --seeds 5`, **5 of 5** do. With `--noise ou`, 2 of 3 seeds end at 1000 (seed 1 learns and then degrades). The point of the seed count is that a single learning curve is one sample. The figure shows the learning curve, the critic's mean Q against its largest possible value 1/(1-γ) = 100, the critic over the force at three pole angles with the actor's choice marked, and the actor as a map over pole angle and angular velocity.

## Lab 4: DDPG fails to explore, and how to fix it

```bash
python lab4_ddpg_mountaincar_continuous.py                 # both runs, about 2 minutes
python lab4_ddpg_mountaincar_continuous.py --run naive --sigma 0.5     # louder, still independent
python lab4_ddpg_mountaincar_continuous.py --run fixed --theta 1.0     # OU with no memory
python lab4_ddpg_mountaincar_continuous.py --run fixed --warmup 0
python play.py --lab 4                                     # the car that learned (--variant naive: the one that did not)
```

The agent is imported from Lab 3 without any change. Only the environment and the noise settings differ.

Expected (seed 0): the **naive** run (Gaussian noise, sigma 0.1) never reaches the flag in training (0 of 30 episodes), visits only positions between -0.78 and -0.20, its actor outputs a force of about 0.005, and its greedy return is exactly 0. The **fixed** run (Ornstein-Uhlenbeck noise, sigma 0.5, theta 0.15) reaches the flag in the 2nd episode and in 170 of 171 training episodes, and its greedy return is about 89 to 93 from step 12,500 on; the missing points are the energy cost -0.1 a² of a nearly full-force policy. The same fixed settings work for seeds 1 and 2. The naive settings fail for seeds 0 and 1, also with sigma 0.5 (Gaussian), and OU with theta 1.0 (white noise) fails too, while theta 0.15 works. What matters is that the noise is correlated in time, not how large it is.

## Design notes

* **Same car, three times.** Labs 1, 2 and 4 use the same hill, so the only differences between them are the ones the lab is about.
* **What `terminated` and `truncated` mean.** The replay buffer stores `terminated` only. A time-out is a limit of the experiment, not a property of the world, so the learning target still bootstraps through it. Lab 3, exercise 4 asks students to break this on purpose.
* **Targets are constants.** In every lab the target `y` is computed under `torch.no_grad()` and the gradient does not flow into it. Compare with the semi-gradient discussion in `rl_course_1h` Lab 3.
* **DDPG and TD3 are one class.** The three TD3 repairs are the flags `twin`, `delay` and `smooth` of the `DDPG` class in Lab 3, each a few lines, so students can read exactly what each one changes.
* **Evaluation is separate.** Evaluation uses its own environment instance and seeds, no exploration noise, and never touches the training data.
* **CSV logs.** Each run writes a CSV in `labs/outputs/` with one row per episode (return, noise, loss, mean Q) and one per evaluation, so a curve can be re-plotted or inspected without rerunning.
* **Checkpoints.** `lab2_dqn.pt`, `lab3_ddpg.pt`, `lab4_fixed.pt`, `lab4_naive.pt` are the files `play.py` loads by default. They are small and are shipped so that `play.py` works before any training.
* **Rendering.** Without `--window`, `play.py` writes an mp4 and opens nothing. With `--window` on macOS, the MuJoCo pendulum needs `mjpython play.py --lab 3 --window`.

## Slides

`RL_Bridge_DQN_DDPG.pptx` (25 slides, with speaker notes) explains the algorithms of the four labs and shows the figures the labs produce. It is built by `build/build_deck.py` from the PNGs in `labs/outputs/`, so run the labs first:

```bash
pip install python-pptx pillow numpy
python build/build_deck.py
python build/check_layout.py RL_Bridge_DQN_DDPG.pptx     # rough check that text fits its boxes
```

## Where this leads

DDPG is off-policy: it reuses old data from a replay buffer. PPO in `SO101_PickCube` is on-policy: it uses each batch of data once and throws it away, and needs no critic over actions and no target networks. Students arriving there already know what a critic, a bootstrapped target and exploration noise are, and can ask what PPO does about each.
