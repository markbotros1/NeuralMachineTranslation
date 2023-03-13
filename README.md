NeuralMachineTranslation
==============================

An LSTM-based Recurrent Neural Network that translates Modern English into Old (Shakespearean) English

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data files used to train and test model
    |
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── data_processing.py
    │   │   └── vocab.py
    │   │
    │   ├── model          <- Scripts to build and train models
    │   │   ├── encoder.py
    │   │   ├── decoder.py
    │   │   ├── nmt.py
    │   │   ├── train_helpers.py
    │   │   ├── train_model.py
    │   │   └── predict_model.py
            
