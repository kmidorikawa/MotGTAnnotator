import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageTk

MAX_WIDTH = 800
MAX_HEIGHT = 600
ANNOTATION_FILE = "annotations.json"

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi Object Tracking Annotator")

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.rectangles = []
        self.current_rect = None
        self.start_x = self.start_y = 0

        self.image_list = []
        self.image_index = 0
        self.image = None
        self.photo = None
        self.image_scale = 1

        self.annotations = {}
        self.folder_path = ""

        self.load_folder_button = tk.Button(root, text="Load Folder", command=self.load_folder)
        self.load_folder_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_button.pack(side=tk.LEFT)

        self.prev_button = tk.Button(root, text="Previous Image", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(root, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(side=tk.LEFT)

        self.load_button = tk.Button(root, text="Load Annotations", command=self.load_annotations)
        self.load_button.pack(side=tk.LEFT)

        self.current_image_label = tk.Label(root, text="Current Image: None")
        self.current_image_label.pack(side=tk.TOP)

        self.image_info_label = tk.Label(root, text="Image 0 of 0")
        self.image_info_label.pack(side=tk.TOP)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.on_right_button_press)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()
            self.image_index = 0
            self.load_image()
            self.adjust_window_size()
            self.load_annotations_from_file()

    def load_image(self):
        if self.image_list:
            image_path = self.image_list[self.image_index]
            self.image = cv2.imread(image_path)

            # Resize image if it exceeds the maximum dimensions
            height, width = self.image.shape[:2]
            self.image_scale = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1)
            if self.image_scale < 1:
                new_size = (int(width * self.image_scale), int(height * self.image_scale))
                self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_AREA)

            self.photo = self.convert_to_photoimage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.load_annotations_for_current_image()
            self.update_labels()

    def adjust_window_size(self):
        if self.image_list:
            image_path = self.image_list[self.image_index]
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            self.image_scale = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1)
            new_size = (int(width * self.image_scale), int(height * self.image_scale))
            self.root.geometry(f"{new_size[0]}x{new_size[1]}")

    def update_labels(self):
        if self.image_list:
            current_image_name = os.path.basename(self.image_list[self.image_index])
            self.current_image_label.config(text=f"Current Image: {current_image_name}")
            self.image_info_label.config(text=f"Image {self.image_index + 1} of {len(self.image_list)}")

    def convert_to_photoimage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(image)
        return self.photo

    def on_button_press(self, event):
        if self.image is None:
            return
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        if self.image is None:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if self.image is None:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        rect = (self.start_x, self.start_y, end_x, end_y)
        default_id = str(len(self.rectangles) + 1)
        while self.id_exists(default_id):
            default_id = str(int(default_id) + 1)
        obj_id = simpledialog.askstring("Input", "Enter Object ID:", initialvalue=default_id)
        if not self.id_exists(obj_id):
            color = self.get_color_for_id(obj_id)
            self.rectangles.append((rect, obj_id, color))
            self.current_rect = None
            self.redraw_rectangles()
        else:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

    def id_exists(self, obj_id):
        for _, id, _ in self.rectangles:
            if id == obj_id:
                return True
        return False

    def get_color_for_id(self, obj_id):
        color_dict = {
            "1": "red",
            "2": "blue",
            "3": "green",
            # Add more color mappings as needed
        }
        return color_dict.get(obj_id, "black")

    def on_right_button_press(self, event):
        if self.image is None:
            return
        rect_to_delete = None
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for rect, obj_id, color in self.rectangles:
            if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                rect_to_delete = (rect, obj_id, color)
                break
        if rect_to_delete:
            self.rectangles.remove(rect_to_delete)
            self.redraw_rectangles()

    def redraw_rectangles(self):
        self.canvas.delete("all")
        if self.photo:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        for rect, obj_id, color in self.rectangles:
            self.canvas.create_rectangle(rect, outline=color)
            coords_text = f"{int(rect[0]/self.image_scale)},{int(rect[1]/self.image_scale)},{int(rect[2]/self.image_scale)},{int(rect[3]/self.image_scale)}"
            self.canvas.create_text((rect[0], rect[1]), text=f"ID: {obj_id} Coords: {coords_text}", anchor=tk.NW, fill=color)

    def next_image(self):
        if self.image_list and self.image_index < len(self.image_list) - 1:
            self.save_annotations_for_current_image()
            self.image_index += 1
            self.load_image()
            self.adjust_window_size()
            self.save_annotations()

    def prev_image(self):
        if self.image_list and self.image_index > 0:
            self.save_annotations_for_current_image()
            self.image_index -= 1
            self.load_image()
            self.adjust_window_size()
            self.save_annotations()

    def save_annotations_for_current_image(self):
        if self.image_list:
            image_path = self.image_list[self.image_index]
            original_rectangles = [(int(r[0] / self.image_scale), int(r[1] / self.image_scale), int(r[2] / self.image_scale), int(r[3] / self.image_scale), obj_id) for r, obj_id, _ in self.rectangles]
            self.annotations[image_path] = original_rectangles

    def load_annotations_for_current_image(self):
        if self.image_list:
            image_path = self.image_list[self.image_index]
            self.rectangles = [(self.scale_rect(r[:4]), r[4], self.get_color_for_id(r[4])) for r in self.annotations.get(image_path, [])]
            self.redraw_rectangles()

    def scale_rect(self, rect):
        return (
            rect[0] * self.image_scale,
            rect[1] * self.image_scale,
            rect[2] * self.image_scale,
            rect[3] * self.image_scale
        )

    def save_annotations(self):
        if self.folder_path:
            annotation_file = os.path.join(self.folder_path, ANNOTATION_FILE)
            with open(annotation_file, 'w') as f:
                json.dump(self.annotations, f)

    def load_annotations(self):
        if self.folder_path:
            annotation_file = os.path.join(self.folder_path, ANNOTATION_FILE)
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                self.load_annotations_for_current_image()

    def load_annotations_from_file(self):
        annotation_file = os.path.join(self.folder_path, ANNOTATION_FILE)
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
            self.load_annotations_for_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotatorApp(root)
    root.mainloop()
