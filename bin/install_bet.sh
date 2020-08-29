#!/bin/sh
# export GIT_ACCOUNT=mathematicalmichael
# export CLONE_BRANCH=sample
# export REPO_NAME=bet
# cd /tmp && \
#     git clone --single-branch --branch ${CLONE_BRANCH} https://github.com/${GIT_ACCOUNT}/${REPO_NAME}.git --depth=1 && \
#     cd ${REPO_NAME} && \
#     pip install .
#
# echo "Installed BET!"
pip install bet==2.3.0
