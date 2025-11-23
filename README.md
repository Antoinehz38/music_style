# Installation

I recommend creating a virtual environment and installing the dependencies using:

```bash
pip install .
```

You can download the data using the following lines :

```commandline
mkdir -p data && wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O data/fma_small.zip
```

```commandline
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -O data/fma_metadata.zip
```

```commandline
unzip data/fma_small.zip -d data/ 
```

```commandline
unzip data/fma_metadata.zip -d data/
```