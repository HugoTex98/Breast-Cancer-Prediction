# Use the official Python 3.12.4 image as the base
FROM python:3.12.4-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the default command to run your Python script
CMD ["python", "scripts/main.py"]
