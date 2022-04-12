FROM python:3.8-buster

# Copy ta-lib installation script
COPY ./install-ta-lib.sh /app/install-ta-lib.sh

# Install ta-lib
RUN sh /app/install-ta-lib.sh

# Copy requirements.txt
COPY ./requirements.txt /app/requirements.txt

# Install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest
COPY ./ /app

RUN python /app/src/bots/test_bot.py