import cv2
import PIL.ImageTk as imtk
import PIL.Image as im
from PIL import Image, ImageOps
import torch
import numpy as np
import locale
import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as msg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from indeterminatebeam import (
    Support,
    Beam,
    PointLoad,
    DistributedLoad,
    PointTorque
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import json
from openai import OpenAI
import custom_detector

MODE_GPT = "gpt"
MODE_LOCAL = "local"

def go_home(widget, root):
  widget.destroy()
  return Home(root)

def convert_local_to_solver(local_data):

  length = 0
  if local_data["positions"]:
    length = max([p["value"] for p in local_data["positions"]])

  supports = []
  for s in local_data["supports"]:
    if s[-1] == 1:
      supports.append(("pinned", s[0]))
    elif s[-1] == 2:
      supports.append(("roller", s[0]))
    elif s[-1] == 3:
      supports.append(("fixed", s[0]))

  loads = []
  for l in local_data["loads"]:
    loads.append(("pload", l[4], l[0]))

  return {
    "Length": length,
    "supports": supports,
    "loads": loads
  }

def isfloat(num):
  try:
    float(num)
    return True
  except ValueError:
    return False
  
def entry_check(trash, text, self):
  number = parse_number(text)
  print(trash)
  return

def parse_number(num_string):
  locale.setlocale(locale.LC_ALL, '')
  num_string = num_string.replace(',', '')
  return locale.atof(num_string)

def format_number(num):
  locale.setlocale(locale.LC_ALL, '')
  return float(locale.format_string('%f', num, grouping=True))

def encode_image(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode("utf-8")

def parse_beam_image(image_path):
  processed_path = preprocess_beam_image(image_path)

  image_base64 = encode_image(processed_path)
  system_prompt = """
  You are a structural beam diagram parser.

  Return ONLY valid JSON.

  Sign convention:
  - Downward forces are negative
  - Upward forces are positive
  - Counterclockwise moments are positive

  Units:
  - Length in meters
  - Force in kN
  
  If any value is uncertain, estimate it based on the diagram scale.

  Schema:
  {
    "beam_length": float,
    "supports": [
      {"type": "pinned|roller|fixed", "position": float}
    ],
    "loads": [
      {"type": "point", "magnitude": float, "position": float},
      {"type": "distributed", "magnitude": float, "start": float, "end": float},
      {"type": "moment", "magnitude": float, "position": float}
    ]
  }
  """

  response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0,
    response_format={"type": "json_object"},
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract the beam data from this image."},
          {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
          }
        ]
      }
    ]
  )

  raw_output = response.choices[0].message.content

  try:
    return json.loads(raw_output)
  except json.JSONDecodeError:
    print(raw_output)
    raise ValueError("Model did not return valid JSON")
  
def convert_gpt_to_solver(data):

  converted = {}

  converted["Length"] = data["beam_length"]

  converted["supports"] = [
      (s["type"], s["position"]) for s in data["supports"]
  ]

  loads = []

  for l in data["loads"]:
    if l["type"] == "point":
      loads.append(("pload", l["magnitude"], l["position"]))

    elif l["type"] == "distributed":
      loads.append(("dload", l["magnitude"], l["start"], l["magnitude"], l["end"]))

    elif l["type"] == "moment":
      loads.append(("moment", l["magnitude"], l["position"]))

  converted["loads"] = loads

  return converted

def preprocess_beam_image(input_path, output_path="processed.png"):

  img = cv2.imread(input_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
 
  kernel = np.ones((2,2), np.uint8)
  cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

  pil_img = Image.fromarray(cleaned)
  pil_img.thumbnail((512, 512), Image.LANCZOS)
  padded = ImageOps.pad(pil_img, (512, 512), color="white")

  padded.save(output_path)
  return output_path
    
class Camera():
  def __init__(self, root) -> None:
    self.master = root
    self.master.title("Detector")
    self.frame = tk.Frame(self.master, bg= "white")

    self.frame.place(relheight= 1, relx= 0, relwidth= 1, rely= 0)
    self.c = tk.Canvas(self.frame, width= 1000, height= 600)

class Home:
  def __init__(self, root) -> None:
    self.master = root
    self.master.title("PBS - Home")

    self.frame = tk.Frame(self.master, bg= "white")
    self.frame.place(
      relheight= 1, relx= 0, 
      relwidth= 1, rely= 0)
    
    font = ("DengXian", 32)
    font_button = ("DengXian", 20)
    
    tk.Label(self.frame, text= "Photo Beam Solver", font= font, fg= "#000", bg= "white", justify= tk.CENTER).place(
      relx= 0.25, rely= 0.1,
      relwidth= 0.5, relheight=0.3)

    input_button = ctk.CTkButton(self.frame, font= font_button,text= "Input your data", command= self.go_BeamSettings)
    image_button = ctk.CTkButton(self.frame, font= font_button,text= "Load a picture", command= self.go_Load_Picture)

    center = 0.4; main_button_width = 0.2; main_button_height = 0.07
    
    input_button.place(
      relx= center, rely= 0.5,
      relheight= main_button_height, relwidth= main_button_width)    
    image_button.place(
      relx= center, rely= 0.61,
      relheight= main_button_height, relwidth= main_button_width)
  
  def go_BeamSettings(self):
    self.frame.place_forget()
    return BeamSettings(self.master)
  
  def go_Load_Picture(self):
    self.frame.place_forget()
    return Load_Picture(self.master)
  
class BeamSettings:
  def __init__(self, master, data=None, image_path=None, mode=MODE_LOCAL):
    self.master = master
    self.master.title("PhotoBeamSolver")

    self.frame = tk.Frame(self.master, bg= "white")
    self.frame.place(
      relheight= 1, relx= 0, 
      relwidth= 1, rely= 0)
    
    self.font_button = ("Consolas", 20, "italic")
    self.font_number = ("Consolas", 20)
    self.font_units_ = ("Consolas", 20)
    self.entry_color = "#EFF9FF"
    
    if image_path is not None:
      detector = BeamDetector(mode=mode)
      raw = detector.detect(image_path)

      if mode == MODE_GPT:
        data = convert_gpt_to_solver(raw)
      else:
        data = convert_local_to_solver(raw)
    
    if data == None:
      data = {
        "Length": 500,
        "supports": [("pinned", 0), ("fixed", 500)],
        "loads": [("pload", 5000, 250), ("dload", 50, 0, 50, 500), ("pload", 100, 0), ("pload", 100, 500)]
        }
      self.data = data
    else:
      self.data = data

    ctk.CTkButton(self.frame, text= "←", text_color= "white", command= lambda: go_home(self.frame, self.master) ).place(
      relx= 0.01, rely= 0.01,
      relwidth= 0.025, relheight= 0.04)
    
    ctk.CTkButton(self.frame, font= self.font_number, text= "Continue", text_color= "white", command= lambda: self.read_continue() ).place(
      relx= 0.1, rely= 0.4,
      relwidth= 0.15, relheight= 0.05)
    
    """
    BEAM PROPERTIES DISPLAY

      E = Elasticity Modulus
      A = Cross Sectional Area
      I = Moment of Inertia
      L = Beam Length
    """

    self.E_var = tk.StringVar()
    self.A_var = tk.StringVar()
    self.I_var = tk.StringVar()
    self.L_var = tk.StringVar()

    self.E_units_var = tk.StringVar()
    self.A_units_var = tk.StringVar()
    self.I_units_var = tk.StringVar()
    self.L_units_var = tk.StringVar()

    self.E_var.set(f"{2100000:,}")
    self.A_var.set(f"{1600:,}")
    self.I_var.set(f"{213333.33:,}")
    self.L_var.set(self.data["Length"])

    self.E_units_var.set("kg/cm2")
    self.A_units_var.set("cm2")
    self.I_units_var.set("cm4")
    self.L_units_var.set("cm")

    E_units_values = ["kg/cm2", "lb/in2"]
    A_units_values = ["cm2", "in2"]
    I_units_values = ["cm4", "in4"]
    L_units_values = ["cm", "in"]

    E_mdlus_entry = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.E_var)
    A_areas_entry = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.A_var)
    I_inert_entry = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.I_var)
    L_legth_entry = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.L_var)

    E_mdlus_units = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= E_units_values, variable= self.E_units_var)
    A_areas_units = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= A_units_values, variable= self.A_units_var)
    I_inert_units = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= I_units_values, variable= self.I_units_var)
    L_legth_units = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= L_units_values, variable= self.L_units_var)

    E_mdlus_label = ctk.CTkLabel(self.frame, font= self.font_button, text= "E")
    A_areas_label = ctk.CTkLabel(self.frame, font= self.font_button, text= "A")
    I_inert_label = ctk.CTkLabel(self.frame, font= self.font_button, text= "I")
    L_legth_label = ctk.CTkLabel(self.frame, font= self.font_button, text= "L")

    self.x_entries = 0.1
    self.y_entries = 0.1
    self.entry_width = 0.15
    self.entry_heght = 0.05
    self.label_width = 0.05
    self.units_width = 0.09

    E_mdlus_entry.place(
      relx= self.x_entries, rely= self.y_entries, 
      relheight= self.entry_heght, relwidth= self.entry_width)
    A_areas_entry.place(
      relx= self.x_entries, rely= self.y_entries + self.entry_heght * 1.2,
      relheight= self.entry_heght, relwidth= self.entry_width)
    I_inert_entry.place(
      relx= self.x_entries, rely= self.y_entries + self.entry_heght * 2.4,
      relheight= self.entry_heght, relwidth= self.entry_width)
    L_legth_entry.place(
      relx= self.x_entries, rely= self.y_entries + self.entry_heght * 3.6,
      relheight= self.entry_heght, relwidth= self.entry_width)
    
    E_mdlus_label.place(
      relx= self.x_entries - self.label_width * 1.1, rely= self.y_entries,
      relwidth= self.label_width, relheight= self.entry_heght)
    A_areas_label.place(
      relx= self.x_entries - self.label_width * 1.1, rely= self.y_entries + self.entry_heght * 1.2,
      relwidth= self.label_width, relheight= self.entry_heght)
    I_inert_label.place(
      relx= self.x_entries - self.label_width * 1.1, rely= self.y_entries + self.entry_heght * 2.4,
      relwidth= self.label_width, relheight= self.entry_heght)
    L_legth_label.place(
      relx= self.x_entries - self.label_width * 1.1, rely= self.y_entries + self.entry_heght * 3.6,
      relwidth= self.label_width, relheight= self.entry_heght)
    
    E_mdlus_units.place(
      relx= self.x_entries + self.entry_width * 1.1, rely= self.y_entries,
      relwidth= self.units_width, relheight= self.entry_heght)
    A_areas_units.place(
      relx= self.x_entries + self.entry_width * 1.1, rely= self.y_entries + self.entry_heght * 1.2,
      relwidth= self.units_width, relheight= self.entry_heght)
    I_inert_units.place(
      relx= self.x_entries + self.entry_width * 1.1, rely= self.y_entries + self.entry_heght * 2.4,
      relwidth= self.units_width, relheight= self.entry_heght)
    L_legth_units.place(
      relx= self.x_entries + self.entry_width * 1.1, rely= self.y_entries + self.entry_heght * 3.6,
      relwidth= self.units_width, relheight= self.entry_heght)
    
    E_mdlus_entry.bind("<KeyRelease>", lambda e, b= self.E_var.get(), self= self: entry_check(e,b,self))
    A_areas_entry.bind("<KeyRelease>", lambda e, b= self.A_var.get(), self= self: entry_check(e,b,self))
    I_inert_entry.bind("<KeyRelease>", lambda e, b= self.I_var.get(), self= self: entry_check(e,b,self))
    L_legth_entry.bind("<KeyRelease>", lambda e, b= self.L_var.get(), self= self: entry_check(e,b,self))

    """
    SUPPORTS SETTINGS DISPLAY
    """
    self.SUPPORTS = ["pinned", "roller", "fixed"]

    sup_width = 0.085

    pls_button_sup = ctk.CTkButton(self.frame, text= "+", font= ("Arial", 14) ,command= lambda: self.add_rem_support("+"))
    min_button_sup = ctk.CTkButton(self.frame, text= "-", font= ("Arial", 14) ,command= lambda: self.add_rem_support("-"))

    pls_button_sup.place(
      relx= 0.35, rely= 0.45, 
      relwidth= sup_width/3, relheight= 0.0375)
    min_button_sup.place(
      relx= 0.35 + sup_width/3 * 1.1, rely= 0.45, 
      relwidth= sup_width/3, relheight= 0.0375)
    
    self.support_typ_list = list()
    self.support_pos_list = list()
    self.support_uni_list = list()
    self.support_typ_var_list = list()
    self.support_pos_var_list = list()

    for i, k in enumerate(self.data["supports"]):
      support_typ_var = tk.StringVar()
      support_pos_var = tk.StringVar()
      support_typ_var.set(k[0])
      support_pos_var.set(k[1])
      
      self.support_typ_var_list.append(support_typ_var)
      self.support_pos_var_list.append(support_pos_var)

      support_typ = ctk.CTkOptionMenu(self.frame, 
        font= self.font_units_, 
        values= self.SUPPORTS, 
        variable= self.support_typ_var_list[len(self.support_typ_var_list)-1])
      support_pos = ctk.CTkEntry(self.frame, 
        font= self.font_number, 
        fg_color= self.entry_color, 
        textvariable= self.support_pos_var_list[len(self.support_pos_var_list)-1])
      support_uni = ctk.CTkLabel(self.frame, 
        text= "cm",
        font= self.font_units_, 
        anchor= "w")

      self.support_typ_list.append(support_typ)
      self.support_pos_list.append(support_pos)
      self.support_uni_list.append(support_uni)

      support_typ.place(
        relx= self.x_entries - self.label_width * 1.1, rely= 0.5 + 0.05 * i * 1.1, 
        relwidth= self.entry_width * 1 / 1.2, relheight= 0.05)
      support_pos.place(
        relx= self.x_entries - self.label_width * 1.1 + 0.125 * 1.1, rely= 0.5 + self.entry_heght * i * 1.1, 
        relwidth= self.entry_width, relheight= self.entry_heght)
      support_uni.place(
        relx= self.x_entries - self.label_width * 1.1 + self.entry_width * 1 / 1.2 * 1.1 + self.entry_width * 1.1, rely= 0.5 + self.entry_heght * i * 1.1, 
        relwidth= self.units_width, relheight= self.entry_heght)
    
    """
    LOADS SETTINGS
    """
    self.LOADS = ["pload", "dload", "moment"]
    self.load_width = 0.085
    banner_font = ("Arial", 12,)

    pls_button_load = ctk.CTkButton(self.frame, text= "+", font= ("Arial", 14) ,command= lambda: self.add_rem_load("+"))
    min_button_load = ctk.CTkButton(self.frame, text= "-", font= ("Arial", 14) ,command= lambda: self.add_rem_load("-"))

    pls_button_load.place(
      relx= 0.92, rely= 0.51875, 
      relwidth= self.load_width/3, relheight= 0.0375)
    min_button_load.place(
      relx= 0.92 + self.load_width/3 * 1.1, rely= 0.51875, 
      relwidth= self.load_width/3, relheight= 0.0375)
    
    ctk.CTkLabel(self.frame, font= banner_font, text= "Type").place(
      relx= 0.45 , rely= 0.5, 
      relwidth= self.load_width, relheight= 0.05)
    ctk.CTkLabel(self.frame, font= banner_font, text= "|F|_s (kg)").place(
      relx= 0.45 + 1 * self.load_width * 1.1, rely= 0.5, 
      relwidth= self.load_width, relheight= 0.05)
    ctk.CTkLabel(self.frame, font= banner_font, text= "x_s (m)").place(
      relx= 0.45 + 2 * self.load_width * 1.1, rely= 0.5, 
      relwidth= self.load_width, relheight= 0.05)
    ctk.CTkLabel(self.frame, font= banner_font, text= "|F|_f (kg)").place(
      relx= 0.45 + 3 * self.load_width * 1.1, rely= 0.5, 
      relwidth= self.load_width, relheight= 0.05)
    ctk.CTkLabel(self.frame, font= banner_font, text= "x_f (m)").place(
      relx= 0.45 + 4 * self.load_width * 1.1, rely= 0.5, 
      relwidth= self.load_width, relheight= 0.05)

    self.load_info = dict()
    
    self.load_load_typ_var_list = list()
    self.load_load_F_s_var_list = list()
    self.load_load_x_s_var_list = list()
    self.load_load_F_f_var_list = list()
    self.load_load_x_f_var_list = list()

    for i, k in enumerate(self.data["loads"]):
      load_typ_var = tk.StringVar()
      load_F_s_var = tk.StringVar()
      load_x_s_var = tk.StringVar()
      load_F_f_var = tk.StringVar()
      load_x_f_var = tk.StringVar()

      self.load_load_typ_var_list.append(load_typ_var)
      self.load_load_F_s_var_list.append(load_F_s_var)
      self.load_load_x_s_var_list.append(load_x_s_var)
      self.load_load_F_f_var_list.append(load_F_f_var)
      self.load_load_x_f_var_list.append(load_x_f_var)

      self.load_load_typ_var_list[i].set(f"{k[0]}")
      self.load_load_F_s_var_list[i].set(f"{k[1]:,}")
      self.load_load_x_s_var_list[i].set(f"{k[2]:,}")
      
      load_typ = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= self.LOADS, variable= load_typ_var)
      load_F_s = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_F_s_var_list[i])
      load_x_s = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_x_s_var_list[i])
      load_F_f = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_F_f_var_list[i])
      load_x_f = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_x_f_var_list[i])

      load_typ.place(
        relx= 0.45, rely= 0.55 + 0.05 * i * 1.1, 
        relwidth= self.load_width, relheight= 0.05)
      load_F_s.place(
        relx= 0.45 + 1 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * i * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_x_s.place(
        relx= 0.45 + 2 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * i * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_F_f.place(
        relx= 0.45 + 3 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * i * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_x_f.place(
        relx= 0.45 + 4 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * i * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      
      if k[0] == "dload":
        self.load_load_F_f_var_list[i].set(k[3])
        self.load_load_x_f_var_list[i].set(k[4])
      else:
        self.load_load_F_f_var_list[i].set("--")
        self.load_load_x_f_var_list[i].set("--")
        load_F_f.configure(state= "disabled")
        load_x_f.configure(state= "disabled")

      self.load_info[str(i)] = {
        "typ": load_typ_var,
        "F_s": load_F_s_var,
        "x_s": load_x_s_var,
        "F_f": load_F_f_var,
        "x_f": load_x_f_var,
        "load_typ": load_typ,
        "load_F_s": load_F_s,
        "load_x_s": load_x_s,
        "load_F_f": load_F_f,
        "load_x_f": load_x_f,}

    """
    DIAGRAMS
    可能跳 work as a function aside
    """

    self.notebook = tk.Frame(self.frame, bg= "white")
    self.notebook.place(
      relheight= 0.5, relx= 0.45, 
      relwidth= 0.55, rely= 0)
    
    tabs = ttk.Notebook(self.notebook)
    
    self.B_tab = tk.Frame(tabs) 
    self.M_tab = tk.Frame(tabs)
    self.V_tab = tk.Frame(tabs)
    self.d_tab = tk.Frame(tabs)

    tabs.add(self.B_tab, text= "Beam")
    tabs.add(self.M_tab, text= "Moment")
    tabs.add(self.V_tab, text= "Shearing")
    tabs.add(self.d_tab, text= "Deflection")
    tabs.pack(expand= 1, fill= "both")

    self.read_continue()

  
  def solve_Beam(self):
    E = parse_number(self.E_var.get())
    A = parse_number(self.A_var.get())
    I = parse_number(self.I_var.get())
    L = parse_number(self.L_var.get())

    if self.E_units_var.get() == "lb/in2":
      E = E / 2.20462262 / 6.4516
    if self.A_units_var.get() == "in2":
      A = A * 2.54 ** 2
    if self.I_units_var.get() == "in4":
      I = I * 2.54 ** 4
    if self.L_units_var.get() == "in":
      L = L * 2.54

    "SUPPORT REACTIONS READING"
    reactions = []
    for i, k in enumerate(self.support_typ_list):
      type_support = self.support_typ_var_list[i].get()
      valu_support = parse_number(self.support_pos_var_list[i].get())
      if type_support == "pinned":
        reactions.append(Support(valu_support, (1,1,0)))
      if type_support == "roller":
        reactions.append(Support(valu_support, (0,1,0)))
      if type_support == "fixed":
        reactions.append(Support(valu_support, (1,1,1)))
    
    "LOAD REACTION READING"
    loads = []
    loads_loaded = self.load_info.keys()
    for i, k in enumerate(loads_loaded):
      if self.load_info[k]["typ"].get() == "dload":
        force = parse_number(self.load_info[k]["F_s"].get())
        x_s = parse_number(self.load_info[k]["x_s"].get())
        x_f = parse_number(self.load_info[k]["x_f"].get())
        loads.append(DistributedLoad(force, (x_s, x_f), 90))
      elif self.load_info[k]["typ"].get() == "moment":
        moment = parse_number(self.load_info[k]["F_s"].get())
        loc = parse_number(self.load_info[k]["x_s"].get())
        loads.append(PointTorque(moment, loc))
      elif self.load_info[k]["typ"].get() == "pload":
        force = parse_number(self.load_info[k]["F_s"].get())
        loc = parse_number(self.load_info[k]["x_s"].get())
        loads.append(PointLoad(-force, loc, 90))
    
    "DATA FORMAT AND SOLUTION"
    beam = Beam(L)
    results = dict()
    for i in loads:
      beam.add_loads(i)
    for i in reactions:
      beam.add_supports(i)
    
    beam.analyse()
    reactions = beam._reactions
    discrete_lenght = np.linspace(0, L, 1000)
    results["length"] = discrete_lenght
    results["moment"] = list()
    results["shear"] = list()
    results["deflection"] = list()
    for i in discrete_lenght:
      results["moment"].append(beam.get_bending_moment(i))
      results["shear"].append(beam.get_shear_force(i))
      results["deflection"].append(beam.get_deflection(i))
    results.update(reactions)
    return results

  def add_rem_support(self, __):
    if __ == "-":
      self.support_typ_list[-1].destroy()
      self.support_pos_list[-1].destroy()
      self.support_uni_list[-1].destroy()
      self.support_typ_list.pop()
      self.support_pos_list.pop()
      self.support_uni_list.pop()
      self.support_typ_var_list.pop()
      self.support_pos_var_list.pop()
    if __ == "+":
      i = len(self.support_typ_list)
      support_typ_var = tk.StringVar()
      support_pos_var = tk.StringVar()
      support_typ_var.set("pinned")
      support_pos_var.set(int(0))

      self.support_typ_var_list.append(support_typ_var)
      self.support_pos_var_list.append(support_pos_var)
      
      support_typ = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= self.SUPPORTS, variable= self.support_typ_var_list[i])
      support_pos = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.support_pos_var_list[i])
      support_uni = ctk.CTkLabel(self.frame, text= "cm", font= self.font_units_, anchor= "w")

      self.support_typ_list.append(support_typ)
      self.support_pos_list.append(support_pos)
      self.support_uni_list.append(support_uni)
      
      support_typ.place(
        relx= self.x_entries - self.label_width * 1.1, rely= 0.5 + 0.05 * i * 1.1, 
        relwidth= self.entry_width * 1 / 1.2, relheight= 0.05)
      support_pos.place(
        relx= self.x_entries - self.label_width * 1.1 + 0.125 * 1.1, rely= 0.5 + self.entry_heght * i * 1.1, 
        relwidth= self.entry_width, relheight= self.entry_heght)
      support_uni.place(
        relx= self.x_entries - self.label_width * 1.1 + self.entry_width * 1 / 1.2 * 1.1 + self.entry_width * 1.1, rely= 0.5 + self.entry_heght * i * 1.1, 
        relwidth= self.units_width, relheight= self.entry_heght)
      
  def add_rem_load(self, __):
    numbers_load = len(self.load_info.keys()) - 1
    if __ == "-":
      self.load_info[str(numbers_load)]["load_typ"].destroy()
      self.load_info[str(numbers_load)]["load_F_s"].destroy()
      self.load_info[str(numbers_load)]["load_x_s"].destroy()
      self.load_info[str(numbers_load)]["load_F_f"].destroy()
      self.load_info[str(numbers_load)]["load_x_f"].destroy()
      del self.load_info[str(numbers_load)]
    
    if __ == "+":
      load_typ_var = tk.StringVar()
      load_F_s_var = tk.StringVar()
      load_x_s_var = tk.StringVar()
      load_F_f_var = tk.StringVar()
      load_x_f_var = tk.StringVar()

      self.load_load_typ_var_list.append(load_typ_var)
      self.load_load_F_s_var_list.append(load_F_s_var)
      self.load_load_x_s_var_list.append(load_x_s_var)
      self.load_load_F_f_var_list.append(load_F_f_var)
      self.load_load_x_f_var_list.append(load_x_f_var)
      
      self.load_load_typ_var_list[numbers_load + 1].set(f"pload")
      self.load_load_F_s_var_list[numbers_load + 1].set(f"{5000:,}")
      self.load_load_x_s_var_list[numbers_load + 1].set(f"{250:,}")
      
      load_typ = ctk.CTkOptionMenu(self.frame, font= self.font_units_, values= self.LOADS, variable= self.load_load_typ_var_list[numbers_load + 1])
      load_F_s = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_F_s_var_list[numbers_load + 1])
      load_x_s = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_x_s_var_list[numbers_load + 1])
      load_F_f = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_F_f_var_list[numbers_load + 1])
      load_x_f = ctk.CTkEntry(self.frame, font= self.font_number, fg_color= self.entry_color, textvariable= self.load_load_x_f_var_list[numbers_load + 1])
      
      load_typ.place(
        relx= 0.45, rely= 0.55 + 0.05 * (numbers_load + 1) * 1.1, 
        relwidth= self.load_width, relheight= 0.05)
      load_F_s.place(
        relx= 0.45 + 1 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * (numbers_load + 1) * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_x_s.place(
        relx= 0.45 + 2 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * (numbers_load + 1) * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_F_f.place(
        relx= 0.45 + 3 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * (numbers_load + 1) * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      load_x_f.place(
        relx= 0.45 + 4 * self.load_width * 1.1, rely= 0.55 + self.entry_heght * (numbers_load + 1) * 1.1, 
        relwidth= self.load_width, relheight= self.entry_heght)
      
      self.load_info[str(numbers_load + 1)] = {
        "typ": load_typ_var,
        "F_s": load_F_s_var,
        "x_s": load_x_s_var,
        "F_f": load_F_f_var,
        "x_f": load_x_f_var,
        "load_typ": load_typ,
        "load_F_s": load_F_s,
        "load_x_s": load_x_s,
        "load_F_f": load_F_f,
        "load_x_f": load_x_f,}
    
  def plotter(self, x, y, data, typef= None):
    fig, ax = plt.subplots(1, tight_layout= True)
    print(data)
    if typef == "reactions":
      # simple support
      size = max(x["length"]) / 50
      rel_coords_x = np.array([-1*size, 0, 1*size, -1*size])
      rel_coords_y = np.array([-1*size, 0, -1*size, -1*size])
      # roller support
      theta = np.arange(0, 360, 1)
      rel_polar_y = size * 0.5 * np.sin(theta * 2 * np.pi / 360)
      rel_polar_x = size * 0.5 * np.cos(theta * 2 * np.pi / 360)
      # fixed support
      rel_coords_x_f = np.array([-0.5*size, -0.5*size, 0.5*size, 0.5*size, -0.5*size])
      rel_coords_y_f = np.array([-0.5*size, 0.5*size, 0.5*size, -0.5*size, -0.5*size])
      # moment
      moment_theta = np.array([i for i in range(180, 360)] + [i for i in range(0, 91)])
      mom_x = size * np.cos(moment_theta * 2 *np.pi / 360)
      mom_y = size * np.sin(moment_theta * 2 *np.pi / 360)
      m_arrow_x = np.array([size / 4, 0, size / 4])
      m_arrow_y = np.array([size + size / 4, size, size - size / 4])
      # pforce
      #x = np.array([0,0])
      y = np.array([])
      
      for i in x.keys():
        if isfloat(i) == True:
          print(i, x[i])
          if x[i][1] != 0:
            if x[i][1] > 0:
              ax.plot([i,i], [-40, -15], "red")
              ax.plot([i-5,i], [-20, -15], "red")
              ax.plot([i+5,i], [-20, -15], "red")
            else:
              ax.plot([i,i], [40, 15], "red")
              ax.plot([i-5,i], [20, 15], "red")
              ax.plot([i+5,i], [20, 15], "red")
          if x[i][2] != 0:
            if x[i][2] > 0:
              ax.plot(mom_x + i, mom_y, color= "red")
              ax.plot(m_arrow_x + i, m_arrow_y, color= "red")
            else: 
              ax.plot(-mom_x + i, mom_y, color= "red")
              ax.plot(-m_arrow_x + i, m_arrow_y, color= "red")

      for i in data["supports"]:
        print(i)
        if i[0] == "pinned":
          ax.plot(rel_coords_x + i[1], rel_coords_y, color= "purple")
        elif i[0] == "roller":
          ax.plot(rel_polar_x + i[1] + max(x["length"]), rel_polar_y - size/2, color= "purple")
        elif i[0] == "fixed":
          ax.plot(rel_coords_x_f + i[1], rel_coords_y_f, color= "purple")

      ax.plot([0, max(x["length"])], [0,0], color= "black", linewidth= 5)
      ax.set_ylim(-50, 50)
      ax.set_yticks([])
      ax.set_xlabel("$x$ [cm]")
      return fig
    
    elif typef == "m":
      ax.plot(x,y, color= "#A7C7E7")
      ax.set_ylabel("$M$ [kg · cm]")
      ax.fill_between(x,y, np.zeros(len(y)), alpha= 0.2, color= "#A7C7E7")
    elif typef == "v":
      ax.plot(x,y, color= "#A7C7E7")
      ax.set_ylabel("$V$ [kg]")
      ax.fill_between(x,y, np.zeros(len(y)), alpha= 0.2, color= "#A7C7E7")
    elif typef == "d":
      ax.plot(x,y, color= "#A7C7E7")
      ax.set_ylabel("$\delta$ [mm]")
      ax.fill_between(x,y, np.zeros(len(y)), alpha= 0.2, color= "#A7C7E7")
    else:
      raise 
    ax.set_xlabel("$x$ [cm]")
    return fig

  def read_continue(self):
    solution = self.solve_Beam()

    fig_resp = self.plotter(solution, None, self.data, typef= "reactions")
    fig_moment = self.plotter(solution["length"], solution["moment"], self.data, typef= "m")
    fig_shear = self.plotter(solution["length"], solution["shear"], self.data, typef= "v")
    fig_deflection = self.plotter(solution["length"], solution["deflection"], self.data, typef= "d")

    self.B_diagram = FigureCanvasTkAgg(fig_resp, master= self.B_tab)
    self.M_diagram = FigureCanvasTkAgg(fig_moment, master= self.M_tab)
    self.V_diagram = FigureCanvasTkAgg(fig_shear, master= self.V_tab)
    self.d_diagram = FigureCanvasTkAgg(fig_deflection, master= self.d_tab)
    self.B_diagram.draw()
    self.M_diagram.draw()
    self.V_diagram.draw()
    self.d_diagram.draw()
    self.B_diagram.get_tk_widget().place(
      relx= 0, rely= 0,
      relheight= 1, relwidth= 1)
    self.M_diagram.get_tk_widget().place(
      relx= 0, rely= 0,
      relheight= 1, relwidth= 1)
    self.V_diagram.get_tk_widget().place(
      relx= 0, rely= 0,
      relheight= 1, relwidth= 1)
    self.d_diagram.get_tk_widget().place(
      relx= 0, rely= 0,
      relheight= 1, relwidth= 1)
    return 0

class Load_Picture:
  def __init__(self, master, data= None):
    self.master = master
    self.master.title("PhotoBeamSolver")

    self.frame = tk.Frame(self.master, bg= "white")
    self.frame.place(
      relheight= 1, relx= 0, 
      relwidth= 1, rely= 0)
    
    self.frame_within = tk.Frame(self.master, bg= "#FBFBFB", bd= 2)
    self.frame_within.place(
      relheight= 0.7, relx= 0.15, 
      relwidth= 0.7, rely= 0.15)
    
    self.font_button = ("Consolas", 20, "italic")
    self.font_number = ("Consolas", 20)
    self.font_units_ = ("Consolas", 20)
    self.entry_color = "#EFF9FF"
    
    self.mode = MODE_LOCAL

    self.mode_selector = ctk.CTkOptionMenu(
        self.frame,
        values=["Local model", "GPT vision"],
        command=self.set_mode
    )

    self.mode_selector.place(
        relx=0.5-0.15,
        rely=0.8,
        relwidth=0.3,
        relheight=0.05
    )

    if data == None:
      data = {
        "Length": 500,
        "supports": [("pinned", 0), ("pinned", 500)],
        "loads": [("pload", 5000, 250), ("dload", 0, 0, 50, 500), ("pload", 100, 0), ("pload", 100, 500)]
        }

    ctk.CTkButton(self.frame, text= "←", text_color= "white", command= lambda: go_home(self.frame, self.master) ).place(
      relx= 0.01, rely= 0.01,
      relwidth= 0.025, relheight= 0.04)
    
    self.open_pic = ctk.CTkButton(self.frame_within, text= "Select file", font= self.font_number, text_color= "white", command= lambda: self.find_pic())
    self.open_pic.place(
      relx= 0.5-0.1, rely= 0.45,
      relwidth= 0.2, relheight= 0.1)

  def set_mode(self, choice):
    if choice == "Local model":
      self.mode = MODE_LOCAL
    else:
      self.mode = MODE_GPT

  def find_pic(self):
    image_path = fd.askopenfilename()
    self.image_path = image_path
    stuff = image_path.split(".")
    if stuff[-1] in ["png", "jpg", "jpeg"]:
      self.fig_resp, self.ax_resp = plt.subplots(1, tight_layout= True)
      img = mpimg.imread(image_path)
      self.imgplot = plt.imshow(img)
      self.image_show = FigureCanvasTkAgg(self.fig_resp, master= self.frame_within)
      self.image_show.draw()
      self.image_show.get_tk_widget().place(
        relx= 0, rely= 0,
        relheight= 1, relwidth= 1)
      self.cont_button = ctk.CTkButton(self.frame, text= "Continue", font= self.font_number, text_color= "white", command= lambda: self.detect_continue())
      self.cncl_button = ctk.CTkButton(self.frame, text= "Cancel", font= self.font_number, text_color= "white", command= lambda: self.ah_no_se_cancela_todo())
      
      self.cont_button.place(
        relx= 0.5-0.2, rely= 0.9,
        relwidth= 0.1, relheight= 0.05)
      self.cncl_button.place(
        relx= 0.5+0.1, rely= 0.9,
        relwidth= 0.1, relheight= 0.05)
    else:
      msg.showwarning("Not a valid file", "Please, select a file with extension .png, .jpg or .jpeg.")

  def detect_continue(self):
    return BeamSettings(master=self.master, image_path=self.image_path, mode=self.mode)
  
  def ah_no_se_cancela_todo(self):
    self.cont_button.destroy()
    self.cncl_button.destroy()
    self.image_show.get_tk_widget().pack_forget() 
    return
  
def detect_local(image_path):
  result = custom_detector.detector(image_path)
  return result

def detect_gpt(image_path):
  data = parse_beam_image(image_path)
  return data

class BeamDetector:
  def __init__(self, mode="local"):
    self.mode = mode

  def detect(self, image_path):
    if self.mode == "local":
      return detect_local(image_path)
    if self.mode == "gpt":
      return detect_gpt(image_path)

if "__main__" == __name__:
  import os 
  mode = MODE_LOCAL
  
  ctk.set_appearance_mode("light")
  ctk.set_default_color_theme("blue")

  root = tk.Tk()
  root.resizable(True, True)
  root.geometry("1000x600")
  root.minsize(width= 750, height= 500)

  client = OpenAIclient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  
  canvas = Home(root)

  root.mainloop()