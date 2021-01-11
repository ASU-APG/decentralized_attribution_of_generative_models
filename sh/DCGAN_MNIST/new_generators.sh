dataset=MNIST
lr=0.0005
batch_size=16
key_iter=10
lp_type=2
beta1=0.5
GAN_type=DCGAN
num_workers=4

#Change This Variables
out_file=step_1_and_2.out
tensorboard_folder=runs
alpha=10
how_many=5

touch $out_file
chmod 777 $out_file

for i in $(seq 2 ${how_many}); do
  experiment="g${i}_k${i}_crop_b0_lp2"
  python new_generator_training.py --experiment $experiment --num_workers $num_workers --GAN_type $GAN_type --dataset $dataset --batch_size $batch_size --key_iter $key_iter --lr $lr --beta1 $beta1 --lp_type $lp_type --alpha $alpha --tensorboard_folder $tensorboard_folder | tee -a ${out_file}
done
