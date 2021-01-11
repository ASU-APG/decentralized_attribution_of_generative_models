model=cycle_gan
key_iter=5
lp_type=2
lr=0.0005
batch_size=1


#Change This Variables
out_file=step_1_and_2.out
tensorboard_folder=runs
dataroot=datasets/cityscapes/
name=cityscapes_photo2label_pretrained

#dataroot=datasets/horse2zebra/
#name=horse2zebra_pretrained

GAN_type=CycleGAN

alpha=1000


touch $out_file
chmod 777 $out_file


python step_1_and_2.py --experiment g1_k1_crop_b0_lp2 --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}


