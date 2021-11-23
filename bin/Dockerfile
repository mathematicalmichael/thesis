FROM docker.io/ubuntu:focal-20201106
# FROM docker.io/ubuntu:impish-20211102
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ=America/Denver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    texlive-base \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \ 
    texlive-science \
    latexmk \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get autoremove -y

CMD ['bash']
