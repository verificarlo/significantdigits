FROM verificarlo/fuzzy:v0.5.0-lapack-python3.8.5-numpy-scipy

COPY . /opt/build/significantdigits

ENV VFC_BACKENDS libinterflop_ieee.so

RUN cd /opt/build/significantdigits && \
    pip3 install . -r requirements_docker.txt

ENV PYTHONPATH=/opt/build/:${PYTHONPATH}

ENV VFC_BACKENDS "libinterflop_mca.so -m rr"

RUN cd /opt/build/significantdigits/significantdigits/ && \
    python3 -m pytest -k fuzzy --nsamples=30 --capture=tee-sys

ENV VFC_BACKENDS libinterflop_ieee.so

RUN cd /opt/build/significantdigits/significantdigits/ && \
    python3 -m pytest -k significant --capture=tee-sys

RUN cd /opt/build/significantdigits/significantdigits/ && \
    python3 -m pytest -k contributing --capture=tee-sys

RUN head -n 1000 /opt/build/significantdigits/significantdigits/*.csv

VOLUME [ "/workdir" ]
WORKDIR "/workdir"