
import os 
import numpy as np
import matplotlib.image as mpimg
from time import time
import math


BLOCK_SIZE = 256

def process_img(img_path, key):
	print(img_path, key)

	# 将图像转换成数组
	image_array = mpimg.imread(img_path)
	image_array = np.expand_dims(image_array,axis=0) # 维度扩展
	print(image_array.shape)

	if key == 2:
		label_array = [ 0.,  0.,  1.,  0.,  0.,  0.]
	elif key == 3:
		label_array = [ 0.,  0.,  0.,  1.,  0.,  0.]
	elif key == 1:
		label_array = [ 0.,  1.,  0.,  0.,  0.,  0.]
	elif key == 0:
		label_array = [ 1.,  0.,  0.,  0.,  0.,  0.]
	elif key == 4:
		label_array = [ 0.,  0.,  0.,  0.,  1.,  0.]
	elif key == 5:
		label_array = [ 0.,  0.,  0.,  0.,  0.,  1.]
	
	return (image_array, label_array)


if __name__ == '__main__':
	path = 'train_data'
	files = os.listdir(path) # 列出当前目录下所有的子目录和文件
	turns = math.ceil(len(files) / BLOCK_SIZE)
	print('{0}个文件分成{1}个批次进行转换处理'.format(len(files),turns))
	for turn in range(turns):
		train_labels = np.zeros((1,6),'float')
		train_imgs = np.zeros([1,120,160,3])

		BLOCK_files = files[turn*BLOCK_SIZE:(turn+1)*BLOCK_SIZE]
		for file in BLOCK_files:
			if not os.path.isdir(file) and file[len(file)-3:len(file)] == 'jpg':
				try:
					key = int(file[0])
					image_array,label_array = process_img(path+'/'+file, key)
					train_imgs = np.vstack((train_imgs,image_array))
					train_labels = np.vstack((train_labels,label_array))
				except Exception as e:
					print('process error:{0}'.format(e))

		train_imgs = train_imgs[1:]
		train_labels = train_labels[1:]
		npz_name = str(int(time()))
		directory = 'train_data_npz'

		if not os.path.exists(directory):
			os.mkdir(directory)
		try:
			np.savez(directory + '/' + npz_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)
		except IOError as e:
			print(e)	

