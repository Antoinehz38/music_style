# Installation

I recommend creating a virtual environment and installing the dependencies using:

```bash
python3 -m venv .venv
source .venv/bin/activate  
pip install --upgrade pip
pip install .
```

You can download the data using the following lines :

```bash
mkdir -p data && wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O data/fma_small.zip
```

```bash
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -O data/fma_metadata.zip
```

```bash
unzip data/fma_small.zip -d data/ 
```

```bash
unzip data/fma_metadata.zip -d data/
```

Finally you have to prepocess your data to match the mels 128 version, use this script : 

```bash
python3 -m src.tools.precompute_mels
```
# BASELINE

## How to reproduce 
The baseline we propose to beat is exposed in the folder baseline, it uses the SmallCNN in [baseline_cnn.py](src/tools/CNNs/baseline_cnn.py)

For the baseline we train using the script [train.py](src/train.py) to train : 

```bash
python -m src.train --run_from "src/runs_configs/baseline.json" --baseline true
```

and to see training logs : 

```bash
tensorboard --logdir=./src/runs --port=6006 
```
Be careful, the training is super long especially if you don't have GPU like me ( it was 48 mins for the training) 

To see the results, how the baseline perform you can run

```bash
 python -m src.eval_model --run_from "src/runs_configs/baseline.json" --baseline true
```
## Results

The baseline has a 58,125 % accuracy which is good but not great, let's try to improve it 


# REMOTE GPU

If you want to use a remote gpu, just download the mels128 data and the metadata as private dataset on kaggle and 
you can use the jupiter notebook [music-style.ipynb](music-style.ipynb) to launch your setup 