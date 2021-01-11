dataset=MNIST
experiment=g1_k1_crop_b0_lp2
lr=0.001
batch_size=128
key_iter=10
lp_type=2
beta1=0.5
GAN_type=DCGAN
num_workers=4

#Change This Variables
how_many_key=4
out_file=key_training.out
tensorboard_folder=runs
alpha=10


touch $out_file
chmod 777 $out_file


#Side Experiment
python new_key_generation.py --how_many_key $how_many_key --num_workers $num_workers --dataset $dataset --batch_size $batch_size --key_iter $key_iter --tensorboard_folder $tensorboard_folder --GAN_type $GAN_type --experiment $experiment | tee -a ${out_file}



