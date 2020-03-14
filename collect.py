import io
import picamera
import car_control
import threading
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'
import pygame
import numpy as np
import picamera.array
from time import ctime,time, sleep

global train_labels, train_img, is_capture_running, key

class SplitFrames(object):
	def __init__(self):
		self.frame_num = 0
		self.output = None

	def write(self, buf):
		global key
		if buf.startswith(b'\xff\xd8'):
			# Start of new frame; close the old one (if any) and
			# open a new output
			if self.output:
				self.output.close()
			self.frame_num += 1
			self.output = io.open('{}_image{}.jpg'.format(key, time()), 'wb')
		self.output.write(buf)


def pi_capture():
	global train_labels, train_img, is_capture_running, key
	# init the train_label array
	print('start capture')
	is_capture_running = True

	with picamera.PiCamera(resolution=(160,120), framerate=30) as camera:
		# 根据实际情况，镜头是否需要的上下翻转
		#camera.vflip = True
		camera.start_preview()
		# Give the camera some warm-up time
		time.sleep(2)
		output = SplitFrames()
		start = time()
		camera.start_recording(output, format='mjpeg')
		camera.wait_recording(120)
		camera.stop_recording()
		finish = time()
	print("Captured {} frames in {} seconds".format(output.frame_num, finish - start))
	print('quit capture')
	is_capture_running = False


def car_drive():
	global is_capture_running,key
	key = 4
	pygame.init()
	pygame.display.set_mode((1,1))
	car_control.car_stop()

	while is_capture_running:
		# 获取人输入的驾驶指令
		events = pygame.event.get()
		for evt in events:
			if evt.type == pygame.KEYDOWN:
				key_input = pygame.key.get_pressed()
				print(key_input[pygame.K_w],key_input[pygame.K_a],key_input[pygame.K_d])
				if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
					print('Forward')
					key = 2
					car_control.car_move_forward()
				elif key_input[pygame.K_a]:
					print('Left')
					car_control.car_turn_left()
					sleep(0.1)
					key = 0
				elif key_input[pygame.K_d]:
					print('Right')
					car_control.car_turn_right()
					sleep(0.1)
					key = 1	
				elif key_input[pygame.K_s]:
					print('Backward')
					car_control.car_move_backward()
					sleep(0.1)
					key = 3
				elif key_input[pygame.K_q] or key_input[pygame.K_e]:
					print("Barricade")
					car_control.car_stop()
					key = 5
			elif evt.type == pygame.KEYUP:
				key_input = pygame.key.get_pressed()
				if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
					print('Forward')
					key = 2
					car_control.car_turn_straight()
					car_control.car_move_forward()
				elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
					print('Backward')
					key = 3
					car_control.car_move_backward()	
				else:
					print('Stop')
					car_control.car_stop()
				# car_control.clean_GPIO()		
	car_control.clean_GPIO()


if __name__ == '__main__':
	global train_labels, train_img, is_capture_running, key

	print('Capture Thread')
	print('-' * 50)
	capture_thread = threading.Thread(target=pi_capture,args=())
	capture_thread.setDaemon(True)  # 随主线程一起结束
	capture_thread.start()

	car_drive()

	while is_capture_running:
		pass

	print('Done!')
	car_control.car_stop()
	car_control.clean_GPIO()
