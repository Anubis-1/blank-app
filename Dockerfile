# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit
RUN pip install streamlit

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for Streamlit to listen on port 8080
ENV PORT 8080

# Run Streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

# docker buildx build --platform linux/amd64 -t registry.gitlab.com/anul-dev-projects/pricelist-tool:latest --push .
#caprover deploy --appName pricelist-tool --imageName registry.gitlab.com/anul-dev-projects/pricelist-tool:latest