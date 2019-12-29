FROM mathematicalmichael/python-tflow-pro:latest
WORKDIR /tmp
RUN cd /tmp && \
    git clone --single-branch --branch sample https://github.com/mathematicalmichael/BET.git --depth=1 && \
    cd BET && \
    pip install .
