model=cycle_gan
key_iter=5
lp_type=2
lr=0.0005
batch_size=1
GAN_type=CycleGAN

#Change This Variables
tensorboard_folder=runs
out_file=step_1_and_2.out
dataroot=datasets/cityscapes/
name=cityscapes_photo2label_pretrained
#dataroot=datasets/horse2zebra/
#name=horse2zebra_pretrained

alpha=1000


touch $out_file
chmod 777 $out_file

for i in $(seq 2 5); do
  experiment="g${i}_k${i}_crop_b0_lp2"
  python new_generator_training.py --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done

