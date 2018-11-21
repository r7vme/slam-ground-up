from python:3.6

RUN apt update && \
    apt install -y cmake \
                   libglu1-mesa \
                   libgl1-mesa-glx

ADD requirements.txt /
RUN pip3 install -r /requirements.txt
