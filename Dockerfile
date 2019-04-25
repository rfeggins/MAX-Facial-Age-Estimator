FROM codait/max-base:v1.1.1

# Fill in these with a link to the bucket containing the model and the model file name
ARG model_bucket=http://max-assets.s3.us.cloud-object-storage.appdomain.cloud/facial-age-estimator/1.0
ARG model_file=assets.tar.gz

RUN wget -nv --show-progress --progress=bar:force:noscroll ${model_bucket}/${model_file} --output-document=assets/${model_file} && \
  tar -x -C assets/ -f assets/${model_file} -v && rm assets/${model_file}

#opencv
RUN apt-get update && apt-get install libgtk2.0 -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

COPY . /workspace

EXPOSE 5000

CMD python app.py
