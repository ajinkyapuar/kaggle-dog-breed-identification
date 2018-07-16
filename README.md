# kaggle-dog-breed-identification

## Transfer Learning 
`IMAGE_SIZE=224`

### MobileNet

`ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"`

python -m scripts.retrain \
--bottleneck_dir=tf_files/bottlenecks \
--model_dir=tf_files/models/"${ARCHITECTURE}" \
--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
--output_graph=tf_files/breedsNet.pb \
--output_labels=tf_files/breedsNet.txt \
--architecture="${ARCHITECTURE}" \
--image_dir=data/dog_breeds

- Final test accuracy = 79.1% (N=1037)

#### Inception

`ARCHITECTURE="inception_v3"`

python -m scripts.retrain \
--bottleneck_dir=tf_files/bottlenecks \
--model_dir=tf_files/models/"${ARCHITECTURE}" \
--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
--output_graph=tf_files/breedCeption.pb \
--output_labels=tf_files/anglesCeption.txt \
--architecture="${ARCHITECTURE}" \
--image_dir=data/dog_breeds

- Final test accuracy = 90.3% (N=1037)
