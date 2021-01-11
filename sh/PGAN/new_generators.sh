dataset=celeba_cropped
batch_size=16
key_iter=5
lr=0.0005
beta1=0.0
lp_type=2
GAN_type=PGAN

#Change This Variables
alpha=100
tensorboard_folder=runs
out_file=naive_new_generators.out
num_workers=4
how_many=5

touch $out_file
chmod 777 $out_file


for i in $(seq 2 ${how_many}); do
  experiment="g${i}_k${i}_crop_b0_lp2"
  python new_generator_training.py --experiment $experiment --num_workers $num_workers --GAN_type $GAN_type --dataset $dataset --batch_size $batch_size --key_iter $key_iter --lr $lr --beta1 $beta1 --lp_type $lp_type --alpha $alpha --tensorboard_folder $tensorboard_folder | tee -a ${out_file}
done