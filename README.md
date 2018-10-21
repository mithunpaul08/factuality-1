# factuality

This is the repository of the factuality prediction project. This repository currently has only one folder containing code of re-implementing Rudinger et al. (2018)’s 2-layer stacked bidirectional linear LSTM model using dynet 2.0.3 instead of pytorch 0.2.0.

### Software organization:
-	Dockerfile: a docker file to install dependencies and directly run on test data with pre-trained model.
-	factuality_test.py: directly run pre-trained model on test data
-	factuality.py: train the model

### Instruction to run:
1. Download this [repository](https://github.com/Fan-Luo/factuality/archive/master.zip) and unzip it.
2. Before running code file factuality_test.py, download data files [en-ud-test.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-test.conllu) (to train the model with factuality.py,  also download [en-ud-train.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-train.conllu) and [en-ud-dev.conllu](https://github.com/UniversalDependencies/UD_English-EWT/blob/r1.2/en-ud-dev.conllu)) and [UDS-IH2](http://decomp.io/projects/factuality/factuality_eng_udewt.tar.gz) and unzip them into the same directory where the Docker file locates.
3. Download the trained model, including trained.model and tables.txt, to the same directory.
4. Open terminal and change to the directory where all the files downloaded, then run the following script:
    
    *docker build -t luo-cs585-hw3 -f ./Dockerfile .*
