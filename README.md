# factuality

This is the repository of the factuality prediction project. This repository currently has only one folder containing code of re-implementing Rudinger et al. (2018)’s 2-layer stacked bidirectional linear LSTM model using dynet 2.0.3 instead of pytorch 0.2.0.

### Software organization:
The folder `L-biLSTM_2\` has the following files:
-	Dockerfile: a docker file to install dependencies and directly run on test data with pre-trained model.
-	factuality_test.py: directly run pre-trained model on test data
-	factuality.py: train the model

### Software required
python 3, dynet 2.0.3, numpy 1.13.3

### Instruction to run:
1. Download this [repository](https://github.com/Fan-Luo/factuality/archive/master.zip) and unzip it.
2. Before running code file factuality_test.py, download data files [en-ud-test.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-test.conllu) (to train the model with factuality.py,  also download [en-ud-train.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-train.conllu) and [en-ud-dev.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-dev.conllu)) and [UDS-IH2](http://decomp.io/projects/factuality/factuality_eng_udewt.tar.gz) and unzip them into the same directory where the Docker file locates.
3. Download the trained model, including [trained.model](https://drive.google.com/file/d/1ONe9B5DhK2E3QO8L_WSz6CfL68YAO0TX/view?usp=sharing) and [tables.txt](https://drive.google.com/file/d/1kNkwuf6LpTHdBnRds78OMRAvntpylWRF/view?usp=sharing), to the same directory.
4. Open terminal and change to the directory where all the files downloaded, then run the following script:
    
    *docker build -t luo-cs585-hw3 -f ./Dockerfile .*
