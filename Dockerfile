FROM python:3.6.4

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' model-api-user

WORKDIR /opt/model_api

ARG PIP_EXTRA_INDEX_URL
ENV FLASK_APP run.py

# Install requirements, including from Gemfury
ADD ./model_api /opt/model_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/model_api/requirements.txt

RUN chmod +x /opt/model_api/run.sh
RUN chown -R model-api-user:model-api-user ./

USER model-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]