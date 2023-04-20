# Specifies base image and tag
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest
WORKDIR /root
ENV PYTHONPATH=/root

# Copies the trainer code to the docker image.
RUN mkdir /root/trainer
COPY trainer /root/trainer
# Installs additional packages as you need.
RUN mkdir /root/env
COPY env/requirements.txt /root/env
RUN pip3 install -r /root/env/requirements.txt

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.run"]
