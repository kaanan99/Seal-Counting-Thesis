FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Make seal_counting the working directory
WORKDIR /seal_counting

# Copy contents of repo into container
COPY . /seal_counting

# Create directory to contain logs
RUN mkdir /seal_counting/tb_logs

# Install dependencies
RUN pip install -r requirements.txt

EXPOSE 8888
EXPOSE 6006

CMD jupyter lab --allow-root --ip="*"  --NotebookApp.token='' --NotebookApp.password='' & tensorboard --logdir /workspace/tb_logs --bind_all
