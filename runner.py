import os

for i in range(1, 4):
	command = 'python train_with_texttask.py -d newsbias_random_%d_untyped --bases 0 --hidden 16 --l2norm 5e-4 --testing --epochs 200 -lr 1e-2' % i
	print(command)
	os.system(command)