import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
# import json
import threading
import random
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unetppSAM import gen_ans
from unetppSAM_specific import gen_ans_specific
from unetppSAM_directory import gen_ans_directory

# DIRECTORY = ""
# FILE = ""

class Application(tk.Frame):
    def __init__(self, configfile_path, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_GUI(configfile_path)
        self.imagebox = tk.Label(root)
        self.imagebox.place(x=250, y=100)
        # self.imagebox.pack(side='left')
        self.file_path = tk.StringVar()
        self.file_path = "sample.png"
        self.directory_path = tk.StringVar()
        self.directory_path = "sample"
        
    def create_GUI(self, configfile_path):
        configfile = open(configfile_path, "r", encoding="utf-8").read()
        self.config = json.loads(configfile)
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("dark_black.TFrame", background=self.config["background1"][0])
        self.style.configure("light_black.TFrame", background=self.config["background1"][1])
        self.style.configure("dark_black.TButton", background=self.config["background1"][0], foreground = self.config["background1"][2])
        
        self.master.geometry(self.config["window_size"])
        self.master.title(self.config["window_title"])
        left_frame = ttk.Frame(self.master, style="dark_black.TFrame")      #畫面左側底部
        # left_frame.pack(side="left")
        left_frame.pack(side="left", fill='both', expand=False)
        right_frame = ttk.Frame(self.master, style="light_black.TFrame")    #畫面右側底部
        # right_frame.pack(side="left", expand=True)
        right_frame.pack(side="left", fill="both", expand=True)
        
        self.gen_is_show_ans = tk.IntVar(value = 1)         #是否在生成完後直接顯示圖片
        self.gen_is_gen_compare = tk.IntVar(value = 1)      #是否生成對照圖
        self.gen_is_show_compare = tk.IntVar(value = 1)     #是否在生成對照圖完後直接顯示圖片
        
        self.checkbutton_base(left_frame, self.gen_is_show_ans , self.config["gen_is_show_ans_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])             
        self.checkbutton_base(left_frame, self.gen_is_gen_compare , self.config["gen_is_gen_compare_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])       
        self.checkbutton_base(left_frame, self.gen_is_show_compare , self.config["gen_is_show_compare_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])     
        # self.button_base(left_frame, self.config["gen_button_text"], self.gen_ans_img, style="dark_black.TButton")
        # self.button_base(left_frame, "show_image", self.gen_image, style="dark_black.TButton")
        # self.button_base(left_frame, "show_image", self.show_image, style="dark_black.TButton")
        self.quit_button(left_frame, "quit", quit, style="dark_black.TButton")

        self.button_base(left_frame, "select_file", self.select_file, style="dark_black.TButton")
        # self.button_base(left_frame, "show_file_path", self.show_file_path, style="dark_black.TButton")
        self.button_base(left_frame, "file_gen", self.gen_ans_img_specific, style="dark_black.TButton")
        self.button_base(left_frame, "show_file_result", self.show_image_specific, style="dark_black.TButton")
        
        self.button_base(left_frame, "select_directory", self.select_derictory, style="dark_black.TButton")
        # self.button_base(left_frame, "show_directory_path", self.show_directory_path, style="dark_black.TButton")
        self.button_base(left_frame, "directory_gen", self.gen_ans_img_directory, style="dark_black.TButton")
        # self.button_base(left_frame, "show_directory_results", self.show_image_directory, style="dark_black.TButton")


    def checkbutton_base(self, window, var, text, background=None, foreground = None):
        checkbutton = tk.Checkbutton(window, text=text, variable=var, onvalue=1, offvalue=0, background=background, foreground = "#AAA")
        checkbutton.pack()
        checkbutton.select()
    
    def button_base(self, window, text, fun, style=None):
        button = ttk.Button(window, text=text, command=fun, style=style)
        button.pack()
    
    def quit_button(self, window, text, fun, style=None):
        button = ttk.Button(window, text=text, command=fun, style=style)
        button.pack(side='bottom')
    
    def gen_ans_img(self):
        # print("type os config: ", type(self.config["oentheSAM.py"]))
        t = threading.Thread(target=gen_ans, args=(self.config["oentheSAM.py"], int(self.gen_is_show_ans.get()), int(self.gen_is_gen_compare.get()), int(self.gen_is_show_compare.get()),))
        t.daemon = self.config["set_daemon"]
        t.start()

    def show_image(self):
        image_files = [f for f in os.listdir(self.config["oentheSAM.py"]["compare_all_floder"])]
        selected_image = random.choice(image_files)
        image_path = os.path.join(self.config["oentheSAM.py"]["compare_all_floder"], selected_image)
        # print(image_path)
        # img = Image.open(image_path)
        
        image = ImageTk.PhotoImage(file=image_path)
        self.imagebox.config(image=image)
        self.imagebox.image = image


    def gen_ans_img_specific(self):
        # print("type os config: ", type(self.config["oentheSAM.py"]))
        t = threading.Thread(target=gen_ans_specific, args=(self.config["oentheSAM.py"], int(self.gen_is_show_ans.get()), int(self.gen_is_gen_compare.get()), int(self.gen_is_show_compare.get()), self.file_path,))
        t.daemon = self.config["set_daemon"]
        t.start()

    def show_image_specific(self):
        image_name = self.file_path.split("/")
        index = len(image_name) - 1
        image_name = image_name[index]
        image_path = "save_compare_all/" + image_name + "_compare_all.png"
        # print(image_path)
        # img = Image.open(image_path)
        
        image = ImageTk.PhotoImage(file=image_path)
        self.imagebox.config(image=image)
        self.imagebox.image = image

    def gen_ans_img_directory(self):
        # print("type os config: ", type(self.config["oentheSAM.py"]))
        t = threading.Thread(target=gen_ans_directory, args=(self.config["oentheSAM.py"], int(self.gen_is_show_ans.get()), int(self.gen_is_gen_compare.get()), int(self.gen_is_show_compare.get()), self.directory_path,))
        t.daemon = self.config["set_daemon"]
        t.start()

    # def show_image_directory(self):
    #     image_name = self.file_path.split("/")
    #     index = len(image_name) - 1
    #     image_name = image_name[index]
    #     image_path = "save_compare_all/" + image_name + "_compare_all.png"
    #     # print(image_path)
    #     # img = Image.open(image_path)
        
    #     image = ImageTk.PhotoImage(file=image_path)
    #     self.imagebox.config(image=image)
    #     self.imagebox.image = image


    def select_file(self):
        self.file_path = filedialog.askopenfilename()

    def select_derictory(self):
        self.directory_path = filedialog.askdirectory()

    def show_file_path(self):
        print(self.file_path)

    def show_directory_path(self):
        print(self.directory_path)


if __name__ == "__main__":
    import json
    configfile = "json/config.json"
    root = tk.Tk()
    GUI = Application(configfile, master=root)
    # show_image("save_compare_all")
    GUI.mainloop()