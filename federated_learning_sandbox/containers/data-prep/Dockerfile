FROM python:3.10.8-bullseye as wheelbuilder

COPY requirements.txt ./

RUN --mount=type=cache,target=/.cache/pip \
    python -m pip install -U pip setuptools wheel \
 && pip wheel --no-deps --cache-dir=/.cache/pip --wheel-dir /wheels -r requirements.txt


FROM python:3.10.8-slim-bullseye

COPY --from=wheelbuilder /wheels /wheels
RUN python -m venv /venv \
 && . /venv/bin/activate \
 && python -m pip install --no-cache -U pip setuptools wheel \
 && pip install --no-cache /wheels/*

ARG UID=5642
ARG GID=5642

RUN ( getent group $GID >/dev/null || groupadd -r -g $GID localgroup ) \
 && useradd -m -l -s /bin/bash -g $GID -N -u $UID localuser \
 && chown -R $UID:$GID /venv

WORKDIR /home/localuser
USER localuser

COPY --chown=$UID:$GID dataprep_gc.py run_dataprep.sh ./

# This is basically what venv/bin/activate does
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"

CMD ["./run_dataprep.sh", "/input", "/output", "/input/cohort_data.csv"]
