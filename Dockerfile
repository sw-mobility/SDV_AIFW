# Use the official Python image for ARM64 architecture
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-warn-script-location -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Ensure all Python files and scripts are executable
RUN chmod +x /app/*.py

# Expose the port the app runs on
EXPOSE 5002

# Run the application
CMD ["python", "app.py"]
