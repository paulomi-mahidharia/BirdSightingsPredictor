**MR CS6240 Project**

**Getting Started**

- Open the project using IDE - Eclipse or IntelliJ 

This project contains Scala code learning about various classification and prediction techniques and apply it on a problem from e-bird dataset.

**Prerequisites**

- Successful execution of all the program versions require two CSV files labeled.csv.bz2 and unlabeled.csv.bz2 (plain csv or bzip2 compressed)
 under the "input" directory. For eg. **"input/labeled.csv.bz2"**.
 
- install SBT version 0.13.0 or higher.

**File Description**

The directory structure contains two folders "src/main/java":

1) **ModelTrainer** - Trains the model on labeled data and saves it to the file system

2) **Predictor** - Uses the saved model from ModelTrainer to predict labels for unlabeled data


**How to Run?**

Each program can be run in the following modes:

1) standalone

- run "make switch-standalone"
- run "make alone"

2) pseudo 

- run "make switch-pseudo"
- run "make pseudo"

3) cloud

- run "make cloud"

**Version**

1.0.0

**Authors**

Paulomi Paresh Mahidharia


