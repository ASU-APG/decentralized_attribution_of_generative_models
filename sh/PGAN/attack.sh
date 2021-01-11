dataset=celeba_cropped
batch_size=11
key_iter=5
lr=0.0005
beta1=0.0
lp_type=2

#Change
alpha=100
GAN_type=PGAN
num_workers=4
suffix=_crop_b0_lp2
number_of_keys=20



attack_type=Blur
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


attack_type=Crop
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


attack_type=Noise
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


attack_type=Jpeg
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done


attack_type=Combination
out_file=$attack_type'.out'
tensorboard_folder='runs_'$attack_type
touch $out_file
chmod 777 $out_file


for i in $(seq 1 $number_of_keys);do
  experiment=g${i}_k${i}$suffix
  python new_generator_training.py --experiment $experiment --is_adversarial --attack_type ${attack_type} --num_workers $num_workers --GAN_type $GAN_type --dataset ${dataset} --batch_size ${batch_size} --key_iter ${key_iter} --lr ${lr} --beta1 ${beta1} --lp_type ${lp_type} --alpha ${alpha} --tensorboard_folder ${tensorboard_folder}| tee -a ${out_file}
done