FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip using Aliyun mirror
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Install requirements using Aliyun mirror
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

COPY . .

EXPOSE 9000

CMD ["python", "app.py"]