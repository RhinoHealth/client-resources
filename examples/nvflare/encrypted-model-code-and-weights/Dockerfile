# Set base Docker image to be used.
FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Set env vars to be able to run apt-get commands without issues.
ENV LC_ALL="C.UTF-8"
ENV TZ=Etc/UTC

# Install Python 3.12
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    export DEBIAN_FRONTEND=noninteractive \
 && apt-get update -qq \
 && apt-get install -q -y --no-install-recommends -q python3 python3-dev python3-venv

# Set up user and group.
ARG UID=5642
ARG GID=5642
RUN <<"EOF" bash
    set -eu -o pipefail
    if [[ $UID -eq 0 ]]; then
        # Symlink /home/localuser to the root user's home dir
        home_dir="$(getent passwd $UID | cut -d: -f6)"
        ln -s "$home_dir" /home/localuser
    else
        if [[ $UID -ge 1000 ]] && getent passwd $UID >/dev/null; then
            # Delete the existing user
            user_name="$(getent passwd $UID | cut -d: -f1)"
            userdel "$user_name" >/dev/null
        fi
        if [[ $GID -ge 1000  ]] && getent group $GID >/dev/null; then
            # Delete the existing group
            group_name="$(getent group $GID | cut -d: -f1)"
            groupdel "$group_name" >/dev/null
        fi
        getent group $GID >/dev/null || groupadd -r -g $GID localgroup
        useradd -m -l -s /bin/bash -g $GID -N -u $UID localuser
    fi
EOF

# Create and "activate" venv.
ENV VIRTUAL_ENV="/venv"
RUN mkdir $VIRTUAL_ENV \
 && chmod g+s $VIRTUAL_ENV \
 && chown $UID:$GID $VIRTUAL_ENV \
 && python3 -m venv --upgrade-deps $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_ROOT_USER_ACTION=ignore pip install -r /requirements.txt

WORKDIR /home/localuser
USER localuser

# Copy the needed local files (code and otherwise) into the container work directory.
# Note: For directories, have a separate COPY command for each top-level directory.
COPY --chown=$UID:$GID ./app ./app
COPY --chown=$UID:$GID ./infer.py ./meta.json ./
COPY --chown=$UID:$GID ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
RUN chmod +x ./infer.py

ENV PYTHONPATH="/home/localuser/app/custom"
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["./entrypoint.sh"]
