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

For the baseline we train using the script [train.py](src/baseline/train.py) to train : 

```bash
python -m src.baseline.train 
```

and to see training logs : 

```bash
tensorboard --logdir=./src/baseline/runs --port=6006 
```
Be careful, the training is super long especially if you don't have GPU like me ( it was 48 mins for the training) 

To see the results, how the baseline perform you can run

```bash
python -m src.baseline.eval_model
```
## Results

The baseline has a 45.9375 % accuracy which is good but not great, let's try to improve it 

