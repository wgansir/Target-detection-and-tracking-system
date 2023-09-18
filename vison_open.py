import tkinter as tk
from subprocess import Popen
import time

def open_siamcar():

    Popen(["python", "vision_siamcar.py"])
    time.sleep(2)
    root.destroy()  # Close the current window

def open_siammask():

    Popen(["python", "vision_siammask.py"])
    time.sleep(2)
    root.destroy()  # Close the current window


def open_siamrpn():

    Popen(["python", "vision_siamrpn++.py"])
    time.sleep(2)
    root.destroy()  # Close the current window

def open_Track_all():
    Popen(["python", "vision_track_all.py"])
    time.sleep(2)
    root.destroy()  # Close the current window

# 创建主窗口
root = tk.Tk()
root.title("跟踪系统选择")
root.geometry("500x300")

# 添加紫色标题条
title_label = tk.Label(root, text="跟踪系统选择", bg="purple", fg="white", font=("Microsoft YaHei", 20, "bold"))
title_label.pack(fill="x", pady=20)

# 添加按键
siamcar_button = tk.Button(root, text="SiamCAR", command=open_siamcar, font=("Helvetica", 14))
siamcar_button.pack(pady=10)

siamfc_button = tk.Button(root, text="SiamMask", command=open_siammask, font=("Helvetica", 14))
siamfc_button.pack(pady=10)

siamrpn_button = tk.Button(root, text="SiamRPN++", command=open_siamrpn, font=("Helvetica", 14))
siamrpn_button.pack(pady=10)

# siamrpn_button = tk.Button(root, text="Track_all", command=open_Track_all, font=("Helvetica", 14))
# siamrpn_button.pack(pady=10)

# 运行界面主循环
root.mainloop()
