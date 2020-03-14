import numpy as np
from sklearn.model_selection import train_test_split
# keras is a high level wrapper on top of tensorflow (machine learning library)
# The Sequential container is a linear stack of layers
from keras.models import Sequential
# popular optimization strategy that uses gradient descent
from keras.optimizers import Adam, SGD
# to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Model, Input
import glob
import sys

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120,160,3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_data():
	"""将数据全部加载到内存中，自动切分成模型训练集和验证集"""

	# load training data
	image_array = np.zeros((1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
	label_array = np.zeros((1,6),'float')
	training_data = glob.glob('train_data/*.npz')

	# 没有数据就自动退出
	if not training_data:
		print('No training data in directory, quit')
		sys.exit()

	for single_npz in training_data:
		with np.load(single_npz) as data:
			imgs_temp = data['train_imgs']
			labels_temp = data['train_labels']
		image_array = np.vstack((image_array,imgs_temp))
		label_array = np.vstack((label_array,labels_temp))	

	X = image_array[1:]	
	y = label_array[1:]
	print("image array shape:"+str(X.shape))
	print("label array shape:"+str(y.shape))
	print(np.mean(X))
	print(np.var(X))

	# now we can split the data into a training (80), testing(20), and validation set
	# thanks scikit learn
	x_train, x_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=0)

	return x_train, x_valid, y_train, y_valid

def build_model(keep_prob):
			
	model = Sequential()
	# BatchNormalization对数据进行预处理，使其归一化数值转换为(0,1)之间的数值
	model.add(BatchNormalization(input_shape=INPUT_SHAPE))
	# 三个隐藏层
	model.add(Conv2D(24, (5,5), activation='elu', strides=(2,2)))
	model.add(Conv2D(36, (5,5), activation='elu', strides=(2,2)))
	model.add(Conv2D(48, (5,5), activation='elu', strides=(2,2)))

	# model.add(Dropout(0.5))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	# model.add(Dropout(0.3))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(keep_prob))
	model.add(Flatten())
	model.add(Dense(500, activation='elu'))
	# model.add(Dropout(0.1))
	model.add(Dense(250, activation='elu'))
	# model.add(Dropout(0.1))
	model.add(Dense(50, activation='elu'))
	# model.add(Dropout(0.1))
	model.add(Dense(5))
	model.summary()

	return model


def train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, X_train, X_valid, y_train, y_valid):

	# 每一个轮次训练的模型都会存下来，进行比较，挑选最优的一个
	# mode:{min,max,auto} mode的选择，如果monitor是监测val_loss，则min
	# 如果monitor是监测val_acc，则max, auto则根据监测的情况自动调换
	checkpoint = ModelCheckpoint('model-{epoch:03d}-{loss:.4f}.h5',
	monitor='val_loss', verbose=0, save_best_only=True, mode='min')

	# EarlyStopping patience：当early stop
	# 被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
	# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
	early_stop = EarlyStopping(
		monitor='val_loss',
		min_delta=.0005,
		patience=4,
		verbose=1,
		mode='min')
	tensorboard = TensorBoard(
		log_dir='./logs',
		histogram_freq=0,
		batch_size=20,
		write_graph=True,
		write_grads=True,
		write_images=True,
		embeddings_freq=0,
		embeddings_layer_names=None,
		embeddings_metadata=None)
	# calculate the difference between expected steering angle and actual steering angle
	# square the difference
	# add up all those differences for as many data points as we have
	# divide by the number of them
	# that value is our mean squared error! this is what we want to minimize via
	# gradient descent
	# opt = SGD(lr=0.0001)
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])

	# Fits the model on data generated batch-by-batch by a Python generator.
	# The generator is run in parallel to the model, for efficiency.
	# For instance, this allows you to do real-time data augmentation on images on CPU in
	# parallel to training your model on GPU.
	# so we reshape our data into their appropriate batches and train our model simulatenously
	model.fit_generator(
		batch_generator(X_train, y_train, batch_size),
		steps_per_epoch=samples_per_epoch/batch_size,
		epochs=nb_epoch,
		max_queue_size=1,
		validation_data=batch_generator(X_valid, y_valid, batch_size),
		validation_steps=len(X_valid)/batch_size,
		callbacks=[tensorboard, checkpoint, early_stop],
		verbose=2)

#    model.fit(X_train,y_train,samples_per_epoch,nb_epoch,max_q_size=1,X_valid,y_valid,\
#              nb_val_samples=len(X_valid),callbacks=[checkpoint],verbose=1)


def batch_generator(X, y, batch_size):
	"""
	Generate training image give image paths and associated steering angles
	"""
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty([batch_size,6])
	while True:
		i = 0
		for index in np.random.permutation(X.shape[0]):
			# permutation矩阵行列置换
			images[i] = X[index]
			steers[i] = y[index]
			i += 1
			if i == batch_size:
				break
		yield (images, steers)


def main():
	print('-' * 30)
	print('Parameters')
	print('-' * 30)

	data = load_data()

	# 以下参数请自己调整测试
	keep_prob = 0.5
	learning_rate = 0.0001
	nb_epoch = 100
	samples_per_epoch = len(data[0])
	batch_size = 256

	print('keep_prob = {}'.format(keep_prob))
	print('learning_rate = {}'.format(learning_rate))
	print('nb_epoch = {}'.format(nb_epoch))
	print('samples_per_epoch = {}'.format(samples_per_epoch))
	print('batch_size = {}'.format(batch_size))
	print('-' * 30)

	# build model
	model = build_model(keep_prob)
	# train model on data, it saves as model.h5
	train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, *data)


if __name__ == '__main__':
	main()	

"""
使用tensorboard对模型训练进行实时监控，

1.在模型代码中设置其tensorboard 的训练日志目录log_dir，
在模型训练fit()函数使用回调参数callbacks=[tensorboard,]
2.在命令提示符中cd到模型所在目录，启动模型开始训练后，
同时开启另外一个命令提示符中cd到模型所在目录，输入
tensorboard --logdir=日志目录名
3.根据2.中命令提示符提示的链接在网页浏览器中打开链接，
即可启动tensorboard实时监控正在训练的模型
"""	