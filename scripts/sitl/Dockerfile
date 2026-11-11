FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates python3 python3-venv file \
    && rm -rf /var/lib/apt/lists/*
COPY betaflight /opt/betaflight
WORKDIR /opt/betaflight
RUN git config core.autocrlf false && git checkout-index --force --all \
    && test "$(git rev-parse HEAD)" = 79065c96ba0bb5cdc675e67d7093e05dab8b330e \
    && git submodule update --init --depth 1 src/config \
    && make TARGET=SITL ARM_SDK_PREFIX= GCC_VERSION="$(gcc -dumpversion)" GCC_REQUIRED_VERSION="$(gcc -dumpversion)" -j4
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir numpy==2.2.6 pybullet==3.2.7 gymnasium==1.2.2 transforms3d==0.4.2 scipy==1.15.3 matplotlib==3.10.3
COPY gym-pybullet-drones /opt/gym-pybullet-drones
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/work:/opt/gym-pybullet-drones"
WORKDIR /results
