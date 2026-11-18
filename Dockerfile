FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data

# "cancer" is bundled with scikit-learn, so the image runs without any data file.
# For Telco: put telco.csv in data/ before building and run with -e TRUSTAUDIT_DATASET=telco
ENV TRUSTAUDIT_DATASET=cancer
EXPOSE 8000
CMD ["uvicorn", "trustaudit.api:app", "--host", "0.0.0.0", "--port", "8000"]
