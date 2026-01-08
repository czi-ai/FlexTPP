# Transformer for Mixed-Type Event Sequences

Felix Draxler, Yang Meng, Kai Nelson, Lukas Laskowski, Yibo Yang, Theofanis Karaletsos, Stephan Mandt

This repository contains an unofficial implementation of the Transformer model for mixed-type event sequences as described in the paper.

```bibtex
@inproceedings{
    draxler2025transformers,
    title={Transformers for Mixed-type Event Sequences},
    author={Felix Draxler and Yang Meng and Kai Nelson and Lukas Laskowski and Yibo Yang and Theofanis Karaletsos and Stephan Mandt},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=MtwsRjPZhf}
}
```

## Installation

Clone the repository:

```bash
git clone https://github.com/fdraxler/FlexTPP.git
cd FlexTPP
```

We use [`uv`](https://docs.astral.sh/uv/) to manage the environment and handle dependencies. To set up the environment, run:

```bash
uv venv
uv sync
```

You can also install the project manually using `pip`, but make sure to activate your virtual environment first. Then, simply install our repository as a package:

```bash
pip install .
````

Be sure to activate the environment before running any code:

```bash
source .venv/bin/activate
```

## Run experiments

We provide the config files for all experiments in the paper in the `configs/` folder. To run an experiment, use the following command:

```bash
python run.py --config-name <name-of-config-file-in-configs>
```

For example, to run the experiment on the EasyTPP Amazon dataset, use:

```bash
python run.py --config-name easytpp_amazon
```

If you are not logged into [Weights & Biases](https://wandb.ai/), you can pass `trainer.logger.0.offline=True` to the command line to log runs locally, or follow the instructions to log in.
