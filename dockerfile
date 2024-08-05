# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
ENV MODEL_PATH=/mnt/efs/model
ENV TRANSFORMERS_CACHE=/tmp
ENV TMPDIR=/tmp

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /mnt/efs/transformers_cache && chmod -R 777 /mnt/efs/transformers_cache
RUN mkdir -p /tmp && chmod 1777 /tmp
# Copy the rest of the application code into the container
COPY . .

EXPOSE 80

# Run the application
CMD ["uvicorn", "app:tenant_search_api.app", "--host", "0.0.0.0", "--port", "80"]