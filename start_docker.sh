docker build -f ./Dockerfile -t cuda_app .
docker run --gpus all cuda_app