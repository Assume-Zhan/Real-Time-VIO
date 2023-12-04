FROM nvcr.io/nvidia/pytorch:21.08-py3

# Arguments
ARG USERNAME=user

# Update
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
                    sudo \
                    vim \
                    curl \
                    tree \
                    git \
                    pip \
                    libgl1-mesa-glx \
                    libxcb-xinerama0 \
                    libxcb-icccm4 \
                    libxcb-image0 \
                    libxcb-keysyms1 \
                    libxcb-render-util0 \
                    '^libxcb.*-dev' \
                    libx11-xcb-dev \
                    libglu1-mesa-dev \
                    libxrender-dev \
                    libxi-dev \
                    libxkbcommon-dev \
                    libxkbcommon-x11-dev \
                    libsm6

# add user with default bash
RUN adduser --disabled-password --gecos '' --shell /bin/bash ${USERNAME} && \
    adduser ${USERNAME} sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Login with user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Clean apt-cache
RUN sudo apt autoremove -y && \
    sudo apt clean -y 

# Install jupyter
RUN pip install jupyter jupyterlab pandas pyqt6 numpy matplotlib opencv-python

# Command in container
CMD ["/bin/bash"]
