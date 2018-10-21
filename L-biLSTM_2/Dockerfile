# use a container for python 3
FROM python:3

# install dependencies with pip 
RUN pip3 install cmake && \
    pip3 install cython && \
    pip3 install numpy && \
    pip3 install git+https://github.com/clab/dynet#egg=dynet

# copy files to the Docker container
ADD factuality_test.py /
ADD it-happened_eng_ud1.2_07092017.tsv /
ADD en-ud-test.conllu /
ADD trained.model /
ADD tables.txt /
RUN python factuality_test.py