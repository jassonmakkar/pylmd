# Use a base image with Python installed
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e pylmd
