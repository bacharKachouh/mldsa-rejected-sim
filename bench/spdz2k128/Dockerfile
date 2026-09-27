# SPDZ2k (malicious, dishonest majority, arithmetic mod 2^k) rebuilt for a 128-bit ring:
# the minimal circuit's reduction mod q multiplies ~54-bit values by ~31-bit constants.
FROM mldsa-bench:0.4.3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ make automake libtool pkg-config git cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/mp-spdz
RUN echo 'CXX = g++' >> CONFIG.mine \
    && echo 'MY_CFLAGS += -DRING_SIZE=128 -Wno-error' >> CONFIG.mine \
    && make -j12 spdz2k-party.x
