docker run \
   --runtime=nvidia \
   -v "/data2/noses_data/seal_counting_thesis_data/Training, Val, and Test Images":"/seal_counting/Training, Val, and Test Images" \
   -v /data2/noses_data/seal_counting_thesis_data/Data:/seal_counting/seal_detector/Data \
   -v /data2/noses_data/seal_counting_thesis_data/TrainedModels:/seal_counting/TrainedModels \
   -p 8889:8888 \
   -p 8890:6006\
   --name seal_thesis_container \
   seal_counting_thesis
