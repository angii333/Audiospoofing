# Enhancing Noise Robustness in anti-spoofing systems

Aasist-main is the main file which consists the code for the model as well as processing MetricGAN+ and adding noise.
Segan-master conists of the Segan processing. 

# AASIST

This repository contains code for training, validating, and evaluating the AASIST model using the ASVspoof 2019 Logical Access dataset.

## Requirements

Ensure that all dependencies are installed before running the code. Install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Train Model
The training, validation, and evaluation processes are included in main.py. You can train the models using the following commands:

```bash
python main.py --config ./config/AASIST.conf
```

## Add noise
We have two seperate codes which take noisy environments and add it to the data.

```bash
python run_noise.py
```

## MetricGAN+ 
We used a pre-trained model from HuggingFace called Metricgan-plus-voicebank and implemented it to our code. To run the file we use this command:
``` bash
pyhton MetricGAN+Enhancement.py
```

# SEGAN

Contains a pre-trained model for segan. The command to run the code is:

```bash
python test_audio1.py
```


