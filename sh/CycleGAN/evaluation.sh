#Change This Variable
how_many_generator=5


echo ------------Non-Robust Distinguishability------------
python distinguish_metric_paper.py --dataroot=datasets/cityscapes/ --name=cityscapes_photo2label_pretrained --model cycle_gan --no_dropout --GAN_type CycleGAN  --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}


echo ------------Non-Robust Attributability---------------
python confusion_PNG.py --dataroot datasets/cityscapes/ --name cityscapes_photo2label_pretrained --model cycle_gan --no_dropout --GAN_type CycleGAN  --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}


#Pick one of attack type from Blur, Crop, Noise, Jpeg, Combination
#And put attack name in attack_type variable
#attack_type=Blur
#python distinguish_metric_paper.py --dataroot=datasets/cityscapes/ --name=cityscapes_photo2label_pretrained --model cycle_gan --no_dropout --GAN_type CycleGAN  --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator} --attack_type ${attack_type}
#python confusion_PNG.py --dataroot datasets/cityscapes/ --name cityscapes_photo2label_pretrained --model cycle_gan --no_dropout --GAN_type CycleGAN  --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator} --attack_type ${attack_type}


echo ------------Lack of Generation quality---------------
python lack_of_generation_quality.py --dataroot datasets/cityscapes/ --name cityscapes_photo2label_pretrained --gpu_ids 0 --no_dropout --GAN_type CycleGAN  --experiment g1_k1_crop_b0_lp2 --how_many_generator ${how_many_generator}
