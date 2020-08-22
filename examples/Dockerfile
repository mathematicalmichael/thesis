FROM quay.io/fenicsproject/stable:current
USER root

### Install package from source
# bet - dependencies for inverse problem
COPY install_bet.sh /tmp/

# need git to install from source, remove after
RUN apt-get update && \
	apt-get install -y \
 	build-essential \
	git \
	&& sh /tmp/install_bet.sh \
	&& apt-get remove git -y \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
	&& apt-get autoremove -y

# formatting tools for scripts, notebooks
RUN pip install \
	nbconvert \
	mud>=0.0.6
	#pyprind  \
	#autopep8 \
	#black \

WORKDIR /temp

# overwrite default python from fenics container
RUN ln -f /usr/bin/python3 /usr/bin/python && \
	ln -f /usr/local/bin/pip3 /usr/local/bin/pip && \
	rm -rf tmp/*

ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH

# need to overwrite fenics' entrypoint to avoid runtime problems
ENTRYPOINT
CMD ["python3"]
