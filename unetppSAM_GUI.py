import tkinter as tk
import tkinter.ttk as ttk
import json
import threading
import random
from PIL import Image, ImageTk
import os
from unetppSAM import gen_ans


class Application(tk.Frame):
    def __init__(self, configfile_path, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_GUI(configfile_path)
        
    def create_GUI(self, configfile_path):
        configfile = open(configfile_path, "r",encoding="utf-8").read()
        self.config = json.loads(configfile)
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("dark_black.TFrame", background=self.config["background1"][0])
        self.style.configure("light_black.TFrame", background=self.config["background1"][1])
        self.style.configure("dark_black.TButton", background=self.config["background1"][0], foreground = self.config["background1"][2])
        
        self.master.geometry(self.config["window_size"])
        self.master.title(self.config["window_title"])
        left_frame = ttk.Frame(self.master, style="dark_black.TFrame")      #畫面左側底部
        left_frame.pack(side="left", fill='both', expand=False)
        right_frame = ttk.Frame(self.master, style="light_black.TFrame")    #畫面右側底部
        # right_frame.pack(side="left", fill="both", expand=True)
        right_frame.pack(side="left")
        
        self.gen_is_show_ans = tk.IntVar(value = 1)         #是否在生成完後直接顯示圖片
        self.gen_is_gen_compare = tk.IntVar(value = 1)      #是否生成對照圖
        self.gen_is_show_compare = tk.IntVar(value = 1)     #是否在生成對照圖完後直接顯示圖片
        
        self.checkbutton_base(left_frame, self.gen_is_show_ans , self.config["gen_is_show_ans_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])             
        self.checkbutton_base(left_frame, self.gen_is_gen_compare , self.config["gen_is_gen_compare_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])       
        self.checkbutton_base(left_frame, self.gen_is_show_compare , self.config["gen_is_show_compare_text"], background=self.config["background1"][0], foreground = self.config["background1"][2])     
        self.button_base(left_frame, self.config["gen_button_text"], self.gen_ans_img, style="dark_black.TButton")
        self.button_base(left_frame, "show_image", self.show_image, style="dark_black.TButton")

        
    def checkbutton_base(self, window, var, text, background=None, foreground = None):
        checkbutton = tk.Checkbutton(window, text=text, variable=var, onvalue=1, offvalue=0, background=background, foreground = "#AAA")
        checkbutton.pack()
        checkbutton.select()
    
    def button_base(self, window, text, fun, style=None):
        button = ttk.Button(window, text=text, command=fun, style=style)
        button.pack()
        
    def gen_ans_img(self):
        t = threading.Thread(target=gen_ans, args=(self.config["oentheSAM.py"],int(self.gen_is_show_ans.get()),int(self.gen_is_gen_compare.get()),int(self.gen_is_show_compare.get())))
        t.daemon = self.config["set_daemon"]
        t.start()

    def show_image(self):
        print("in show image function")
        image_files = [f for f in os.listdir(self.config["oentheSAM.py"]["compare_all_floder"])]
        selected_image = random.choice(image_files)
        image_path = os.path.join(self.config["oentheSAM.py"]["compare_all_floder"], selected_image)
        print(image_path)
        # img = Image.open(image_path)
        img = Image.open(image_path)
        tk_img = ImageTk.PhotoImage(img)

        label = tk.Label(root, image=tk_img, width=128, height=128)
        label.pack()

        # canvas = tk.Canvas(root, width=256, height=256)
        # canvas.create_image(0, 0, anchor='nw', image=tk_img)
        # canvas.pack()


if __name__ == "__main__":
    configfile = "json/config.json"
    root = tk.Tk()
    GUI = Application(configfile, master=root)
    GUI.mainloop()