ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# 1. Instalar git (CRÍTICO: Lo necesitas para DVC y algunas dependencias)
USER root
RUN apt-get update && apt-get install -y git && apt-get clean

# 2. Configurar usuario y permisos
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

# 3. Copiar requirements e instalar
# IMPORTANTE: Apuntamos a src/requirements.txt
COPY --chown=${KEDRO_UID}:${KEDRO_GID} src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el proyecto
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# 5. Comando por defecto
# Usamos tail -f /dev/null para mantener el contenedor "vivo" y listo para recibir órdenes de Airflow
CMD ["tail", "-f", "/dev/null"]