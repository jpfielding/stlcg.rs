FROM rockylinux/rockylinux:10 AS base

ARG REPO_NAME=stlcg.rs
ARG USERNAME=stlcg
ARG GROUPNAME=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Add ca certs so we can build the image on vpn
COPY cas.pem /etc/pki/ca-trust/source/anchors/custom_cas.pem
RUN update-ca-trust

RUN dnf install -y epel-release && dnf -y update
RUN dnf install -y curl-minimal git iproute iputils jq make openssl sudo wget which zip zsh \
    gcc gcc-c++ pkgconf-pkg-config openssl-devel

FROM base as user

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
  useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
  echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
  chmod 0440 /etc/sudoers.d/${USERNAME}

########### NON ROOT FROM HERE DOWN ###########
USER ${USERNAME}

RUN set -eux && \
    mkdir -p ${HOME}/bin && \
    sudo mkdir -p /workdir/${REPO_NAME} && \
    sudo chown ${USERNAME}:${USERNAME} /workdir

# envs
ENV HOME "/home/${USERNAME}"
ENV SDK_HOME ${HOME}/sdk
ENV PATH ${HOME}/bin:${PATH}

# pretty bash
COPY --chown=${USERNAME}:${USERNAME} .bash_pretty ${HOME}/.bash_pretty
RUN set -eux && \
    echo "source ${HOME}/.bash_pretty" >> ${HOME}/.bashrc

# claude
ENV NODE_EXTRA_CA_CERTS="/etc/pki/ca-trust/source/anchors/cas.pem"
ENV PATH="$HOME/.local/bin:$PATH"

# claude install
RUN curl -fsSL https://claude.ai/install.sh | bash

# rust (matches Cargo.toml rust-version MSRV; stable installed alongside)
ENV RUST_MSRV "1.92.0"
ENV RUSTUP_HOME ${SDK_HOME}/rustup
ENV CARGO_HOME ${SDK_HOME}/cargo
ENV PATH ${CARGO_HOME}/bin:${PATH}
RUN set -eux && \
    mkdir -p ${SDK_HOME} && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --default-toolchain stable \
                  --component rustfmt clippy rust-src rust-analyzer && \
    rustup toolchain install ${RUST_MSRV} --component rustfmt clippy && \
    rustup default stable && \
    cargo --version && rustc --version
