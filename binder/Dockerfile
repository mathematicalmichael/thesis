# this user image already has a TON of thing in it.
# my to-do is to slim it down to a more minimal set of dependencies
FROM mathematicalmichael/math-user:thesis
RUN rm -rf ~/work/
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache nbgitpuller

RUN jupyter serverextension enable --py nbgitpuller --sys-prefix

WORKDIR /home/jovyan/thesis
