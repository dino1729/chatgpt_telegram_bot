FROM python:3.11.4

RUN \
    set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    libyaml-dev \
    python3-venv \
    ffmpeg \
    git \
    ca-certificates \
    libasound2 \
    wget \
    ; \
    rm -rf /var/lib/apt/lists/*

# Download and install OpenSSL
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf - \
    && cd openssl-1.1.1u \
    && ./config --prefix=/usr/local \
    && make -j $(nproc) \
    && make install_sw install_ssldirs

# Update library cache
RUN ldconfig -v

# Set SSL_CERT_DIR environment variable
ENV SSL_CERT_DIR=/etc/ssl/certs

RUN pip3 install -U pip && pip3 install -U wheel && pip3 install -U setuptools==59.5.0
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -r /tmp/requirements.txt

COPY . /code
WORKDIR /code

CMD ["bash"]
