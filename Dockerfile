FROM python:3.9-slim
WORKDIR ntpd-ml-ci-cd
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
