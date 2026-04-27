FROM locustio/locust:latest
COPY locustfile.py /locustfile.py
ENTRYPOINT ["locust", "-f", "/locustfile.py", "--headless"]
