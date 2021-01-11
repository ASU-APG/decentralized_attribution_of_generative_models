key_iter=5
lr=0.0005
lp_type=2
GAN_type=CycleGAN
model=cycle_gan
batch_size=1


#Dataset
dataroot=datasets/cityscapes/
name=cityscapes_photo2label_pretrained
#dataroot=datasets/horse2zebra/
#name=horse2zebra_pretrained

#Change
number_of_keys=5
suffix=_crop_b0_lp2
alpha=1000



attack_type=Combination
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --is_adversarial --attack_type ${attack_type} --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done



attack_type=Blur
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --is_adversarial --attack_type ${attack_type} --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done



attack_type=Crop
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --is_adversarial --attack_type ${attack_type} --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done


attack_type=Noise
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --is_adversarial --attack_type ${attack_type} --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done


attack_type=Jpeg
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --is_adversarial --attack_type ${attack_type} --experiment ${experiment} --gpu_ids 0 --no_dropout --GAN_type $GAN_type --dataroot $dataroot --name $name --model $model --key_iter $key_iter --lp_type $lp_type --tensorboard_folder $tensorboard_folder --lr $lr --alpha $alpha --batch_size $batch_size | tee -a ${out_file}
done
