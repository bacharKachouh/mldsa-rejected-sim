# MP-SPDZ, pinned release, for the open-w1 threshold ML-DSA baseline benchmark.
FROM ubuntu:24.04
ARG MPSPDZ_VERSION=0.4.3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils python3 python3-pip iproute2 openssl libgmp-dev libntl-dev \
        libsodium-dev libssl-dev libboost-dev libboost-thread-dev libboost-filesystem-dev \
        libboost-iostreams-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN curl -sL https://github.com/data61/MP-SPDZ/releases/download/v${MPSPDZ_VERSION}/mp-spdz-${MPSPDZ_VERSION}.tar.xz \
        | tar xJ && mv mp-spdz-${MPSPDZ_VERSION} mp-spdz
WORKDIR /opt/mp-spdz
RUN Scripts/tldr.sh && Scripts/setup-ssl.sh 16
