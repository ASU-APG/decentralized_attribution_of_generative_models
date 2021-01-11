dataset=celeba_cropped
experiment=g1_k1_crop_b0_lp2
batch_size=32
key_iter=10
GAN_type=PGAN

#Change This Variables
out_file=key_generation.out
tensorboard_folder=runs_key_generation
how_many_key=4
num_workers=4

touch $out_file
chmod 777 $out_file


python new_key_generation.py --is_side_experiment --how_many_key $how_many_key --num_workers $num_workers --dataset $dataset --batch_size $batch_size --key_iter $key_iter --tensorboard_folder $tensorboard_folder --GAN_type $GAN_type --experiment $experiment | tee -a ${out_file}
