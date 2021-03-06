# coding = utf-8

import RPi.GPIO as GPIO
import time


BACKWHEELINPUT1 = 15
BACKWHEELINPUT2 = 13

FRONTWHEELINPUT1 = 7
FRONTWHEELINPUT2 = 11

BACKWHEELENABLE = 40
FRONTWHEELENABLE = 36

speed1 = 66
speed2 = 32

GPIO.setmode(GPIO.BOARD)

GPIO.setup(BACKWHEELINPUT1,GPIO.OUT)
GPIO.setup(BACKWHEELINPUT2,GPIO.OUT)
GPIO.setup(FRONTWHEELINPUT1,GPIO.OUT)
GPIO.setup(FRONTWHEELINPUT2,GPIO.OUT)
GPIO.setup(FRONTWHEELENABLE,GPIO.OUT)
GPIO.setup(BACKWHEELENABLE,GPIO.OUT)

BACKWHEELPWM = GPIO.PWM(BACKWHEELENABLE,100)
BACKWHEELPWM.start(0)

def car_stop():
	GPIO.output(BACKWHEELINPUT1,GPIO.LOW)
	GPIO.output(BACKWHEELINPUT2,GPIO.LOW)
	GPIO.output(FRONTWHEELENABLE,GPIO.LOW)
	GPIO.output(FRONTWHEELINPUT1,GPIO.LOW)
	GPIO.output(FRONTWHEELINPUT2,GPIO.LOW)
	
def car_move_forward():
	GPIO.output(BACKWHEELINPUT1,GPIO.HIGH)
	GPIO.output(BACKWHEELINPUT2,GPIO.LOW)
	BACKWHEELPWM.ChangeDutyCycle(speed1)

def car_move_backward():
	GPIO.output(BACKWHEELINPUT2,GPIO.HIGH)
	GPIO.output(BACKWHEELINPUT1,GPIO.LOW)
	BACKWHEELPWM.ChangeDutyCycle(speed2)

def car_turn_left():
	GPIO.output(FRONTWHEELENABLE,GPIO.HIGH)
	GPIO.output(FRONTWHEELINPUT1,GPIO.HIGH)
	GPIO.output(FRONTWHEELINPUT2,GPIO.LOW)

def car_turn_right():
	GPIO.output(FRONTWHEELENABLE,GPIO.HIGH)
	GPIO.output(FRONTWHEELINPUT2,GPIO.HIGH)
	GPIO.output(FRONTWHEELINPUT1,GPIO.LOW)
	
def car_turn_straight():
	GPIO.output(FRONTWHEELENABLE,GPIO.LOW)
	GPIO.output(FRONTWHEELINPUT2,GPIO.LOW)
	GPIO.output(FRONTWHEELINPUT1,GPIO.LOW)	

def clean_GPIO():
	GPIO.cleanup()
	BACKWHEELPWM.stop()


