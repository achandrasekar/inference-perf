FROM grafana/k6:latest
COPY k6_script.js /k6_script.js
ENTRYPOINT ["k6", "run", "/k6_script.js"]
