# Keystroke Authentication

This project focuses on implementing keystroke dynamics biometrics as part of the authentication process.

We are using python [keyboard](https://pypi.org/project/keyboard/) library to record keystrokes and 
[keras](https://keras.io/) to create and train neural network model.

Training and evaluation data are stored in "simple_dataset.csv" file. Data was extracted from
[DSL_StrongPasswordData.csv](https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv) from a
[Comparing Anomaly-Detection Algorithms for Keystroke Dynamics](https://www.cs.cmu.edu/~maxion/pubs/KillourhyMaxion09.pdf)
project by Kevin Killourhy and Roy Maxion.

Authors: Grzegorz Kmita, Michał Moskal

## Install required packages

```
pip3 install -r requirements.txt
```


## Creating model

In order to create model and train it with test data, run:

```
$ python3 neural_network.py
```

Model and central vector will be saved for future use in ./model directory.
Plots will be saved in ./plots directory.

## Registering user

A user needs to be registered in order to create their biometric template. Login, template and passphrase that need to be
rewritten 5 times during registration process are stored as an entry in user_data.pickle.

To register user run:

```
$ python3 register_user.py
```

Registration process will look like that:

```
$ username: [create your username here]
$ PASSPHRASE = [random passphrase will be provided here]
$ (1): [rewrite passphrase here]
$ ...
$ (5): [rewrite passphrase 5 times]
```

## Authenticating user

Registered user can be authenticated. User needs to provide a sample that will
be transformed using neural network model into a vector. Then a similarity between 
the vector and biometric template will be calculated.

To authenticate user run:

```
$ python3 authenticate_user.py
```

Authentication process will look like that:

```
$ username: [create your username here]
$ PASSPHRASE = [random passphrase will be provided here]
$ --> [rewrite passphrase here]
```