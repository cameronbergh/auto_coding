FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
COPY . /app
WORKDIR /app
RUN apt-get install build-essential
RUN pip install -r requirements.txt --ignore-installed
EXPOSE 5000
CMD nvidia-smi
CMD python ./index.py

