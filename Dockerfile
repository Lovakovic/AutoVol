# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV GOOGLE_APPLICATION_CREDENTIALS /gcloud_key/vertex_key.json
ENV PYTHONPATH /app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     git     build-essential     python3-dev     gcc g++ make autoconf automake libtool     zlib1g-dev     libyara-dev     libtsk-dev     cmake     pkg-config     unar     && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel, setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone Volatility 3 into 'vol_src'
RUN git clone --depth 1 --branch v2.26.0 https://github.com/volatilityfoundation/volatility3.git vol_src

# Install Volatility 3 and its [full] dependencies from the cloned source
RUN pip install --no-cache-dir --verbose ./vol_src[full]

# Copy the AutoVol application requirements file
COPY requirements.txt .

# Install AutoVol application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the AutoVol application code into the container
COPY autovol ./autovol

# Define the entry point for the container.
ENTRYPOINT ["python", "-m", "autovol.main"]

# Default command
CMD ["--help"] 

