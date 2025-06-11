FROM python:3.12.1-alpine

WORKDIR /app

# Install build tools and dependencies
RUN apk update && apk add --no-cache \
    build-base \
    gcc \
    g++ \
    libffi-dev \
    musl-dev \
    freetype-dev \
    libpng-dev \
    openblas-dev \
    lapack-dev \
    tcl-dev \
    tk-dev \
    linux-headers \
    bash

# Optional: fix issues with some pip builds
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
