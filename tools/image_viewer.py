"""
四区域图像浏览器（GUI 图像浏览工具）
-------------------------------------
使用 Tkinter 构建一个主窗口，包含四个图像浏览区域（2×2 网格布局）。
每个区域可以独立选择一个图片文件夹，并进行图片切换。
还提供底部的全局控制按钮，统一切换所有区域的图像。

主要功能：
- 支持每区域独立选择文件夹
- 图像自适应区域大小，保持比例不变形
- 全局上一张/下一张按钮控制全部区域同步切图
- 自动窗口居中、美观简洁的 UI 风格

依赖库：
- tkinter（Python 标准库 GUI 工具）
- PIL（通过 Pillow 安装：pip install pillow）
"""


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class ImageViewerFrame(tk.Frame):
    """
    单个图像浏览区域（子窗口）组件。

    功能：
    - 显示图像（自适应缩放，不拉伸）
    - 左右切换当前文件夹中的图像
    - 选择文件夹按钮支持加载多图
    - 内部使用 Canvas 和控制按钮区域
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_paths = []
        self.current_index = 0
        self.tk_img = None
        self.config(bg="#f4f4f4", bd=1, relief=tk.SOLID)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))  # 调整下边距
        self.filename_label = tk.Label(self, text="", font=("Arial", 9), bg="#f4f4f4", fg="#555")
        self.filename_label.pack(padx=10, pady=(2, 5))
        self.build_controls()
        self.bind("<Configure>", lambda e: self.show_image())
        self.canvas.bind("<Configure>", lambda e: self.show_image())
        self.zoom_scale = 1.0
        self.min_scale = 0.2
        self.max_scale = 5.0
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom(+1))
        self.canvas.bind("<Button-5>", lambda e: self.zoom(-1))

    def build_controls(self):
        btn_frame = tk.Frame(self, bg="#f4f4f4")
        btn_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        self.prev_btn = tk.Button(btn_frame, text="←", command=self.show_prev, font=("Arial", 10), bg="#e0e0e0", relief=tk.FLAT, width=6)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = tk.Button(btn_frame, text="→", command=self.show_next, font=("Arial", 10), bg="#e0e0e0", relief=tk.FLAT, width=6)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.select_btn = tk.Button(btn_frame, text="选择文件夹", command=self.select_folder, font=("Arial", 10), bg="#007acc", fg="white", relief=tk.FLAT)
        self.select_btn.pack(side=tk.RIGHT, padx=5)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        if not folder_path: return
        self.image_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ])
        self.current_index = 0
        self.show_image()

    def show_image(self):
        if not self.image_paths:
            self.canvas.delete("all")
            return
        img_path = self.image_paths[self.current_index]
        img = Image.open(img_path)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10: return
        img_ratio = img.width / img.height
        canvas_ratio = canvas_w / canvas_h
        if img_ratio > canvas_ratio:
            new_w = canvas_w
            new_h = int(canvas_w / img_ratio)
        else:
            new_h = canvas_h
            new_w = int(canvas_h * img_ratio)
        scaled_w = int(new_w * self.zoom_scale)
        scaled_h = int(new_h * self.zoom_scale)
        img_resized = img.resize((scaled_w, scaled_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_img, anchor=tk.CENTER)
        self.filename_label.config(text=os.path.basename(img_path))

    def show_prev(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()

    def show_next(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image()
    
    def on_mousewheel(self, event):
        delta = event.delta
        direction = 1 if delta > 0 else -1
        self.zoom(direction)

    def zoom(self, direction):
        new_scale = self.zoom_scale * (1.1 if direction > 0 else 0.9)
        if self.min_scale <= new_scale <= self.max_scale:
            self.zoom_scale = new_scale
            self.show_image()



class App:
    """
    图像浏览器主程序入口类。

    功能：
    - 初始化主窗口并居中显示
    - 创建 2×2 图像浏览区域（ImageViewerFrame）
    - 创建底部全局控制按钮（切换全部图片）
    """
    def __init__(self, root):
        self.root = root
        self.root.title("多窗口图像浏览器 (xfxuezhang.cn)")
        win_w, win_h = 1000, 900
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.configure(bg="#f0f0f0")
        self.viewers = []
        self.build_layout()

    def build_layout(self):
        """
        构建主图像显示区域：2行2列 ImageViewerFrame
        """
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 0))
        for row in range(2):
            main_frame.rowconfigure(row, weight=1, pad=10)
            for col in range(2):
                main_frame.columnconfigure(col, weight=1, pad=10)
                viewer = ImageViewerFrame(main_frame)
                viewer.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
                self.viewers.append(viewer)
        self.build_global_controls()

    def build_global_controls(self):
        """
        创建底部全局按钮：所有上一张 / 所有下一张
        """
        bottom_bar = tk.Frame(self.root, bg="#f0f0f0")
        bottom_bar.pack(fill=tk.X, pady=15)
        inner_frame = tk.Frame(bottom_bar, bg="#f0f0f0")
        inner_frame.pack(anchor="center")
        btn_prev = tk.Button(inner_frame, text="← 所有上一张", font=("Arial", 12), bg="#e0e0e0", relief=tk.FLAT, width=15, command=self.global_prev)
        btn_prev.pack(side=tk.LEFT, padx=20)
        btn_next = tk.Button(inner_frame, text="所有下一张 →", font=("Arial", 12), bg="#e0e0e0", relief=tk.FLAT, width=15, command=self.global_next)
        btn_next.pack(side=tk.LEFT, padx=20)

    def global_prev(self):
        """
        触发所有 ImageViewerFrame 显示上一张图像
        """
        for viewer in self.viewers:
            viewer.show_prev()

    def global_next(self):
        """
        触发所有 ImageViewerFrame 显示下一张图像
        """
        for viewer in self.viewers:
            viewer.show_next()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
