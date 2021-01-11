model=cycle_gan
key_iter=15
lp_type=2
lr=0.0005
lrK=0.001
GAN_type=CycleGAN
experiment=g1_k1_crop_b0_lp2

#Change This Variables
out_file=new_key_training.out
tensorboard_folder=runs_new_key
dataroot=datasets/cityscapes/
name=cityscapes_photo2label_pretrained
#dataroot=datasets/horse2zebra/
#name=horse2zebra_pretrained
how_many_key=4
alpha=1000
batch_size=4

touch $out_file
chmod 777 $out_file


python new_key_generation.py --is_side_experiment --lrK $lrK --how_many_key $how_many_key --gpu_ids 0 --no_dropout --experiment $experiment --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr ${lr} --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
