FROM python:3.10.6-buster
COPY computer_vision /computer_vision
RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -e .
CMD uvicorn computer_vision.api.fast:app --host 0.0.0.0 --port $PORT
