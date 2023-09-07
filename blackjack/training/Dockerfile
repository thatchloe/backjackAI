FROM tensorflow/tensorflow:latest-gpu
COPY training training
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY setup.py setup.py
COPY lewagon-bootcamp-392414-1d50cb6cb1d9.json lewagon-bootcamp-392414-1d50cb6cb1d9.json
COPY data.zip data.zip
ENV DATA_PATH=".data"
ENV BUCKET_NAME="blackjack_le_wagon_final"
ENV GOOGLE_APPLICATION_CREDENTIALS=lewagon-bootcamp-392414-1d50cb6cb1d9.json
RUN pip install --upgrade pip
RUN apt-get update && apt-get install git -y
RUN make prepare_cloud_folders
RUN pip install .
RUN pip install --upgrade git+https://github.com/keras-team/keras-cv
RUN apt-get install \
  'wget'\
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y
CMD python training/main.py
