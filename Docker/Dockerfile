FROM python:3.8.1

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

COPY . /code/

RUN pip install -r requirements.txt 

ENV PYTHONPATH /code

CMD [ "python", "/code/qs_3.py" ]