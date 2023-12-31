# Stage 1: Build Python environment with dependencies
FROM python:3.9-slim AS builder

# Set the working directory to /app
WORKDIR /app

# Copy only the requirements file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt into a virtual environment
RUN apt-get update && apt-get install -y git-all 
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

# Stage 2: Create the final image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the Python environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME RunCustomAIModels

# Use the virtual environment created in stage 1
ENV PATH="/opt/venv/bin:$PATH"

# Run the FastAPI app when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
