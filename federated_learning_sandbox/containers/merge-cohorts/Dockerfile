FROM python:3.10.8-slim-bullseye

# Set up non-root user and group.
ARG UID=5642
ARG GID=5642
RUN ( getent group $GID >/dev/null || groupadd -r -g $GID localgroup ) \
 && useradd -m -l -s /bin/bash -g $GID -N -u $UID localuser

# Create and "activate" venv.
ENV VIRTUAL_ENV="/venv"
RUN mkdir $VIRTUAL_ENV \
 && chmod g+s $VIRTUAL_ENV \
 && chown $UID:$GID $VIRTUAL_ENV \
 && python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
 && rm requirements.txt

WORKDIR /home/localuser
USER localuser

# Copy code.
COPY --chown=$UID:$GID merge_cohorts_data.py ./

ENV PYTHONUNBUFFERED=1

CMD [ "python", "./merge_cohorts_data.py", \
      "--input_dir", "/input", \
      "--output_dir", "/output", \
      "--cohort_csv_file", "/input/cohort_data.csv" \
]
