import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

directory_path = filedialog.askdirectory()
print(directory_path)

file_path = filedialog.askopenfilename()
print(file_path)
