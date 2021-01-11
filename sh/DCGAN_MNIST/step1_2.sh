experiment=g1_k1_crop_b0_lp2
lr=0.0005
key_iter=10
lp_type=2
beta1=0.5
GAN_type=DCGAN
num_workers=4


#Change This Variables
dataset=MNIST
batch_size=16
out_file=step_1_and_2.out
tensorboard_folder=runs
alpha=10

touch $out_file
chmod 777 $out_file

python step_1_and_2.py --num_workers $num_workers --dataset $dataset --alpha $alpha  --batch_size $batch_size --key_iter $key_iter --lr $lr --beta1 $beta1 --lp_type $lp_type --tensorboard_folder $tensorboard_folder --GAN_type $GAN_type --experiment $experiment | tee -a ${out_file}


