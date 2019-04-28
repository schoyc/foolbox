FROM tensorflow/tensorflow:1.12.0-gpu-py3

WORKDIR /build

COPY . /build

RUN pip install -r requirements.txt

ENV PYTHONPATH /notebooks

# EXPOSE 8888

# ADD . /launch
# CMD ["bash"]
