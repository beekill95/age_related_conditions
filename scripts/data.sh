#!/bin/bash

poetry run kaggle competitions download -c icr-identify-age-related-conditions && \
    mkdir data && \
    unzip icr-identify-age-related-conditions.zip -d data/
    
