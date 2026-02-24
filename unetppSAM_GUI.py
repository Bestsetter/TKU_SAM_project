import json
import os
import random
import threading

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from PIL import Image, ImageTk

from unetppSAM import gen_ans, gen_ans_specific, gen_ans_directory


class Application(tk.Frame):
    def __init__(self, configfile_path, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_GUI(configfile_path)
        self.imagebox = tk.Label(root)
        self.imagebox.place(x=250, y=100)
        self.file_path = "sample.png"
        self.directory_path = "sample"

    def create_GUI(self, configfile_path):
        with open(configfile_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("dark_black.TFrame", background=self.config["background1"][0])
        self.style.configure("light_black.TFrame", background=self.config["background1"][1])
        self.style.configure("dark_black.TButton", background=self.config["background1"][0], foreground=self.config["background1"][2])

        self.master.geometry(self.config["window_size"])
        self.master.title(self.config["window_title"])

        left_frame = ttk.Frame(self.master, style="dark_black.TFrame")
        left_frame.pack(side="left", fill='both', expand=False)
        right_frame = ttk.Frame(self.master, style="light_black.TFrame")
        right_frame.pack(side="left", fill="both", expand=True)

        self.gen_is_show_ans = tk.IntVar(value=1)
        self.gen_is_gen_compare = tk.IntVar(value=1)
        self.gen_is_show_compare = tk.IntVar(value=1)

        self.checkbutton_base(left_frame, self.gen_is_show_ans, self.config["gen_is_show_ans_text"], background=self.config["background1"][0])
        self.checkbutton_base(left_frame, self.gen_is_gen_compare, self.config["gen_is_gen_compare_text"], background=self.config["background1"][0])
        self.checkbutton_base(left_frame, self.gen_is_show_compare, self.config["gen_is_show_compare_text"], background=self.config["background1"][0])

        self.button_base(left_frame, "select_file", self.select_file, style="dark_black.TButton")
        self.button_base(left_frame, "file_gen", self.gen_ans_img_specific, style="dark_black.TButton")
        self.button_base(left_frame, "show_file_result", self.show_image_specific, style="dark_black.TButton")

        self.button_base(left_frame, "select_directory", self.select_directory, style="dark_black.TButton")
        self.button_base(left_frame, "directory_gen", self.gen_ans_img_directory, style="dark_black.TButton")

        self.quit_button(left_frame, "quit", quit, style="dark_black.TButton")

    def checkbutton_base(self, window, var, text, background=None):
        cb = tk.Checkbutton(window, text=text, variable=var, onvalue=1, offvalue=0,
                            background=background, foreground="#AAA")
        cb.pack()
        cb.select()

    def button_base(self, window, text, fun, style=None):
        ttk.Button(window, text=text, command=fun, style=style).pack()

    def quit_button(self, window, text, fun, style=None):
        ttk.Button(window, text=text, command=fun, style=style).pack(side='bottom')

    def _inference_config(self):
        return self.config["inference"]

    def _inference_flags(self):
        return (int(self.gen_is_show_ans.get()),
                int(self.gen_is_gen_compare.get()),
                int(self.gen_is_show_compare.get()))

    def gen_ans_img_specific(self):
        flags = self._inference_flags()
        t = threading.Thread(target=gen_ans_specific,
                             args=(self._inference_config(), *flags, self.file_path))
        t.daemon = self.config["set_daemon"]
        t.start()

    def show_image_specific(self):
        image_name = os.path.basename(self.file_path)
        image_path = os.path.join(self.config["inference"]["compare_all_floder"],
                                  image_name + self.config["inference"]["compare_all_img_name"])
        image = ImageTk.PhotoImage(file=image_path)
        self.imagebox.config(image=image)
        self.imagebox.image = image

    def gen_ans_img_directory(self):
        flags = self._inference_flags()
        t = threading.Thread(target=gen_ans_directory,
                             args=(self._inference_config(), *flags, self.directory_path))
        t.daemon = self.config["set_daemon"]
        t.start()

    def select_file(self):
        self.file_path = filedialog.askopenfilename()

    def select_directory(self):
        self.directory_path = filedialog.askdirectory()


if __name__ == "__main__":
    configfile = "json/config.json"
    root = tk.Tk()
    GUI = Application(configfile, master=root)
    GUI.mainloop()
