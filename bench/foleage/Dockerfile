# FOLEAGE PCG reference implementation (Bombar et al., ASIACRYPT 2024), pinned commit, for measuring
# F2 Beaver-triple expansion speed on this machine. Research code; semi-honest PCG.
FROM mldsa-bench:0.4.3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc make git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone --recursive https://github.com/sachaservan/FOLEAGE-PCG.git foleage \
    && cd foleage && git checkout 1a56769 && git submodule update --init --recursive
WORKDIR /opt/foleage
RUN sed -i 's#-I/opt/homebrew/opt/openssl/include##' Makefile && make -j8
