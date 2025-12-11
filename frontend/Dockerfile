# frontend/Dockerfile

# Use Python 3.9 (matches your backend version for consistency)
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# (Added the aliyun mirror since I saw you trying to use it in your logs)
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Copy the rest of the application code
COPY . .

# Expose the Gradio port
EXPOSE 7860


CMD ["python", "app.py"]