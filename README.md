# Using ML on Gaze Data in a Fear Generalization setup

The system performs 1) social anxiety recognition; 2) identification of the subjects; 3) eletroctactile stimulus recognition and 4) shock expectancy rating prediction; based on eye movements modeled as an Ornstein-Uhlenbeck process.

## Installation

Install required libraries with Anaconda:

```bash
conda create --name mlgaze -c conda-forge --file requirements.txt
conda activate mlgaze
```
Install [NSLR-HMM](https://gitlab.com/nslr/nslr-hmm)

```bash
python -m pip install git+https://gitlab.com/nslr/nslr
```

### Features extraction
Extract Ornstein-Uhlenbeck features from [Diagnostic Facial Features & Fear Generalization dataset](https://osf.io/4gz7f/) (`datasets/Reutter`) launching the module `extract_OU_params.py`, results will be saved in `features/Reutter_OU_posterior_VI`.


### Train and test
Modules `kfold_identity.py`,`kfold_rating.py`,`kfold_shock.py` and `kfold_social_anxiety.py` exploit different classifiers for classification and regression on the features extracted as an Ornstein-Uhlenbeck process.