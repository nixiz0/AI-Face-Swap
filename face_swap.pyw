import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from tkinter.constants import *
import os.path
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


PARAMS = {
    "bg": "#587B7F",
    "fg": "#ffffff",
    "font": "-family {Arial Rounded MT Bold} -size 24",
    "highlightbackground": "#d9d9d9",
    "highlightcolor": "black",
}
HIGHLIGHTS = {
    "highlightbackground": "#d9d9d9",
    "highlightcolor": "black",
}

class Menu:
    def __init__(self, top=None):
        top.geometry("380x320+750+184")
        top.minsize(380, 320)
        top.maxsize(400, 380)
        top.title("Face Swapping")
        top.configure(background="#587B7F")
        top.configure(**HIGHLIGHTS)

        self.top = top
        
        self.Label1 = tk.Label(self.top)
        self.Label1.configure(**PARAMS)
        self.Label1.place(relx=0.274, rely=0.085, height=41, width=314)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(anchor='w')
        self.Label1.configure(compound='left')
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(text='''Face-Swap''')
        
        self.image = tk.PhotoImage(file="ressources/face_swap_logo.png")
        self.image_label = tk.Label(self.top, image=self.image, background="#587B7F")
        self.image_label.place(relx=0.31, rely=0.2)

        self.Button1 = tk.Button(self.top)
        self.Button1.configure(**HIGHLIGHTS)
        self.Button1.place(relx=0.274, rely=0.5, height=44, width=177)
        self.Button1.configure(activebackground="#7b7979")
        self.Button1.configure(activeforeground="black")
        self.Button1.configure(background="#B36C24")
        self.Button1.configure(command=two_swap)
        self.Button1.configure(compound='left')
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font="-family {Arial Rounded MT Bold} -size 14")
        self.Button1.configure(foreground="#ffffff")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Two Swap''')
        
        self.Button1_1 = tk.Button(self.top)
        self.Button1_1.configure(**HIGHLIGHTS)
        self.Button1_1.place(relx=0.274, rely=0.68, height=44, width=177)
        self.Button1_1.configure(activebackground="#7b7979")
        self.Button1_1.configure(activeforeground="black")
        self.Button1_1.configure(background="#A7631E")
        self.Button1_1.configure(command=all_swap)
        self.Button1_1.configure(compound='left')
        self.Button1_1.configure(disabledforeground="#a3a3a3")
        self.Button1_1.configure(font="-family {Arial Rounded MT Bold} -size 14")
        self.Button1_1.configure(foreground="#ffffff")
        self.Button1_1.configure(highlightbackground="#d9d9d9")
        self.Button1_1.configure(highlightcolor="black")
        self.Button1_1.configure(pady="0")
        self.Button1_1.configure(text='''All Swap''')

# Functions :
def all_swap():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model('neurons_weigths/inswapper_128.onnx', download=False, download_zip=False)

    def open_file_dialog():
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")])
        return file_path

    messagebox.showinfo("Information", "Veuillez sélectionner l'image par défaut")
    img1_fn = open_file_dialog()
    messagebox.showinfo("Information", "Veuillez sélectionner l'image où les visages seront remplacés par l'image précédente")
    img2_fn = open_file_dialog()

    def swapping_show_all(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True):
        img1 = cv2.imread(img1_fn)
        img2 = cv2.imread(img2_fn)

        if plot_before:
            fig, ax = plt.subplots()
            ax.imshow(img1[:,:,::-1])
            ax.axis('off')
            plt.show()

        face1 = app.get(img1)[0]
        face = app.get(img2)[0]
        faces = app.get(img2)

        img1_customs = app.get(img1)
        img1_custom = img1_customs[0]
        img2_ = img2.copy()
        
        for face in faces:
            img2_ = swapper.get(img2_, face, img1_custom, paste_back=True)
        
        if plot_after:
            fig, ax = plt.subplots()
            ax.imshow(img2_[:,:,::-1])
            ax.axis('off')
            plt.show()

    swapping_show_all(img1_fn, img2_fn, app, swapper)

def two_swap():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    swapper = insightface.model_zoo.get_model('neurons_weigths/inswapper_128.onnx', download=False, download_zip=False)
    
    def open_file_dialog():
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")])
        return file_path

    messagebox.showinfo("Information", "Veuillez sélectionner une image à Swap")
    img1_fn = open_file_dialog()
    messagebox.showinfo("Information", "Veuillez sélectionner la deuxième image à Swap")
    img2_fn = open_file_dialog()

    def swapping_show(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True):
        img1 = cv2.imread(img1_fn)
        img2 = cv2.imread(img2_fn)

        if plot_before:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img1[:,:,::-1])
            axs[0].axis('off')
            axs[1].imshow(img2[:,:,::-1])
            axs[1].axis('off')
            plt.show()

        face1 = app.get(img1)[0]
        face2 = app.get(img2)[0]

        img1_ = img1.copy()
        img2_ = img2.copy()

        if plot_after:
            img1_ = swapper.get(img1_, face1, face2, paste_back=True)
            img2_ = swapper.get(img2_, face2, face1, paste_back=True)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img1_[:,:,::-1])
            axs[0].axis('off')
            axs[1].imshow(img2_[:,:,::-1])
            axs[1].axis('off')
            plt.show()
        return img1_, img2_
    
    swapping_show(img1_fn, img2_fn, app, swapper)


if __name__ == '__main__':
    root = tk.Tk()
    app = Menu(root)
    root.mainloop()