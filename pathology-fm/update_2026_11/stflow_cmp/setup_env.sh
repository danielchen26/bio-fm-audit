#!/bin/sh
# separate environment for running the public STFlow code (nothing is installed into the project's own venv)
cd "$(dirname "$0")" || exit 1
UV=$(command -v uv || echo $HOME/.local/bin/uv)
# the public STFlow code, pinned to the commit every result here was run on
COMMIT=880c2eeaa64aaf64ffcd30f3c3278bdab4ab1dbe
[ -d STFlow ] || { git clone -q https://github.com/Graph-and-Geometric-Learning/STFlow && git -C STFlow checkout -q $COMMIT; }
$UV venv env --python 3.12 -q
$UV pip install -q --python env/bin/python torch torch_geometric scanpy scprep mygene einops timm scikit-learn pandas h5py tqdm wandb scvi-tools
env/bin/python -c "import torch, torch_geometric, scanpy, scprep, einops, timm, sklearn; print('env ok', torch.__version__, 'mps', torch.backends.mps.is_available())"
(cd STFlow && git log -1 --format='%H %cd')
# then, once: env/bin/python patch_train.py && env/bin/python patch_model.py && env/bin/python patch_json.py && env/bin/python make_train_val.py
echo SETUP_DONE
