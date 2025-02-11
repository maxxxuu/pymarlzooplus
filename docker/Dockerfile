FROM python:3.9.20

# Do not prompt during apt-get commands
ARG DEBIAN_FRONTEND=noninteractive

# Ensure output to stdout and stderr are written to the terminal without a buffer
ENV PYTHONUNBUFFERED 1

# Add library dependencies for compiling data science python packages
RUN apt-get update &&  \
    apt-get install --no-install-recommends -y gcc libc-dev build-essential python3-dev vim

# Create a user so that we don't run as root
RUN groupadd -r myuser && useradd -m -r -g myuser myuser
USER root

WORKDIR /home/myuser/

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install ffmpeg libsm6 libxext6  -y  && \
    apt-get install -y libgl1-mesa-dev libosmesa6-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version

RUN apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/apt/lists/*

RUN conda create --channel=conda-forge -n pymarlzooplus python=3.8.18 pip=24.0 wheel=0.38.4 setuptools=65.5.0 einops pyopengl

RUN echo "source activate pymarlzoo_plus" >> ~/.bashrc

RUN curl -o install_torch_scatter.sh https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/install_package/installation/install_torch_scatter.sh && \
    chmod +x install_torch_scatter.sh && \
    /bin/bash -c "source activate pymarlzoo_plus && bash install_torch_scatter.sh"

RUN /bin/bash -c "source activate pymarlzoo_plus && pip install git+https://github.com/AILabDsUnipi/pymarlzooplus.git@install_package#egg=pymarlzoo_plus[all]"

CMD ["source", "activate", "pymarlzoo_plus", "&&", "tail", "-f", "/dev/null"]


