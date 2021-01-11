#Change This Variable
how_many_generator=5


echo ------------Non-Robust Distinguishability------------
python distinguish_metric_paper.py --dataset celeba_cropped --num_workers 4 --batch_size 100 --GAN_type DCGAN --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}


echo ------------Non-Robust Attributability---------------
python confusion_PNG.py --dataset celeba_cropped --GAN_type DCGAN --batch_size 100 --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}

#Uncommnet this part when you evaluate robust training
#Pick one of attack type from Blur, Crop, Noise, Jpeg, Combination
#And put attack name in attack_type variable
#attack_type=Combination
#python distinguish_metric_paper.py --dataset celeba_cropped --num_workers 4 --batch_size 100 --GAN_type DCGAN --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator} --attack_type ${attack_type}
#python confusion_PNG.py --dataset celeba_cropped --GAN_type DCGAN --batch_size 100 --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator} --attack_type ${attack_type}


echo ------------Lack of Generation quality---------------
python lack_of_generation_quality.py --dataset celeba_cropped --GAN_type DCGAN --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}
