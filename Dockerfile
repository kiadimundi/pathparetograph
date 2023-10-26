# Use the official Python image as the base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt

RUN git clone https://github.com/bhargavchippada/forceatlas2.git
RUN pip install forceatlas2/
RUN patch /usr/local/lib/python3.10/site-packages/fa2/forceatlas2.py < difference.patch
RUN pip install --trusted-host pypi.python.org Cython==3.0.3
RUN pip install --trusted-host pypi.python.org numpy==1.25.1
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8050 for the Dash app
EXPOSE 8050

# Run app.py when the container launches
CMD ["python", "app.py"]
