FROM python

WORKDIR /app

COPY . /app

EXPOSE 8180

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "start2.py"]
