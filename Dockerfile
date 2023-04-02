ARG FUNCTION_DIR="/function"

FROM python:3.8

RUN apt-get update &&\
    apt install -y python3-pip &&\
	apt install tzdata -y

ENV TZ="America/Bogota"

COPY ./ /${FUNCTION_DIR}/

WORKDIR /${FUNCTION_DIR}/

RUN pip3 install -r ${FUNCTION_DIR}/requirements.txt --no-cache-dir

EXPOSE 5000

ENTRYPOINT [ "python3" ]


CMD [ "app.py" ]