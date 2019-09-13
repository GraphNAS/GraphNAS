FROM python:3
WORKDIR /GraphNAS
COPY ./requirements.txt /GraphNAS
RUN pip install -r requirements.txt
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl

# pytorch-geometric denpendencies
RUN pip install --verbose --no-cache-dir torch-scatter
RUN pip install --verbose --no-cache-dir torch-sparse
RUN pip install --verbose --no-cache-dir torch-cluster
RUN pip install torch-geometric
