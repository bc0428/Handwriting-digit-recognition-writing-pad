import tkinter as tk
from PIL import ImageGrab
import warnings
# import model
warnings.simplefilter('ignore')
root = None
filename = 'temp'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def predict_handwriting():
    m = tf.keras.models.load_model('handwriting model')

    img = tf.keras.utils.load_img(path='temp.png', color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    with tf.device('/cpu:0'):
        load = tf.keras.Sequential([
            tf.keras.layers.Resizing(28,28),
            tf.keras.layers.Rescaling(scale=1./255)
        ])
        img_array = load(img_array)

    pred = m(img_array)
    pred = np.argmax(pred)

    canvas2.create_text(140, 25, text=str(pred), fill='black', font=('Helvetica 45 bold'))


def make_prediction(event):
    x = root.winfo_rootx() + root.winfo_x() *2
    y = root.winfo_rooty() + root.winfo_y() *2
    width = root.winfo_width()
    height = root.winfo_height()
    ImageGrab.grab().crop((x, y, x+width * 2-20, y+height*2-100)).save(filename+".png")

    predict_handwriting()

def coordinates(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def draw(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y),
                       fill='black',
                       width=25)
    lastx, lasty = event.x, event.y

def delete(event):
    canvas.delete('all')
    canvas2.delete('all')


root = tk.Tk()
root.geometry("280x330")
root.bind("<Return>", make_prediction)

canvas = tk.Canvas(root, bg='white')
canvas.pack(anchor='nw', fill='both', expand=True)
canvas2 = tk.Canvas(root, bg='red')
canvas2.place(x=0, y=280)


canvas.bind("<Button-1>", coordinates)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-2>", delete)

root.mainloop()





