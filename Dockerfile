FROM aoirint/openpose-python

RUN mkdir /code
WORKDIR /code

RUN apt update && apt install -y x11-apps
