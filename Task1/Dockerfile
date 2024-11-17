FROM ubuntu:22.04 AS resume_screener
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3.10 && \
    apt-get install python3-pip git awscli -y && \
    apt-get install -y ffmpeg && \
    apt-get install -y tesseract-ocr
WORKDIR /src
COPY ./src /src

# RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3.10 -m pip install -r /src/requirements.txt
RUN python3.10 -m pip install --upgrade pip
# RUN python3.10 -m pip install --upgrade awscli

# gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8082 main:app
CMD ["gunicorn","--timeout","400", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:6000", "main:app"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "8"]