FROM python:3.12-alpine

WORKDIR /opt/app


RUN apk add --no-cache gcc musl-dev libffi-dev libxml2-dev libxslt-dev

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt --ignore-installed

COPY main.py main.py
COPY pyproject.toml pyproject.toml
COPY app app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
