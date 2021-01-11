dataset=MNIST
lr=0.0005
batch_size=16
key_iter=10
lp_type=2
beta1=0.5
GAN_type=DCGAN

#Change This Variables
alpha=10
num_workers=4
suffix=_crop_b0_lp2
how_many_generator=4


attack_type=Blur
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 ${how_many_generator});do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done

#Cropping
attack_type=Crop
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 ${how_many_generator});do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


#Noise
attack_type=Noise
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 ${how_many_generator});do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


#Jpeg
attack_type=Jpeg
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 ${how_many_generator});do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


#Combination
attack_type=Combination
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 ${how_many_generator});do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done
