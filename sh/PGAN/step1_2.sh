dataset=celeba_cropped
experiment=g1_k1_crop_b0_lp2

lr=0.0005
batch_size=16
key_iter=5
lp_type=2
beta1=0.0
GAN_type=PGAN

#Change This Variables
out_file=step_1_and_2.out
tensorboard_folder=runs
alpha=100
num_workers=4


touch $out_file
chmod 777 $out_file


python step_1_and_2.py --num_workers $num_workers --dataset $dataset --alpha $alpha  --batch_size $batch_size --key_iter $key_iter --lr $lr --beta1 $beta1 --lp_type $lp_type --tensorboard_folder $tensorboard_folder --GAN_type $GAN_type --experiment $experiment | tee -a ${out_file}


