import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import easyocr as eso
import warnings
import re

#from yolov5.models.yolo import Detect
#from yolov5.detect import run

"""
 * 90% overlap is the same
"""

def extract_number_and_units(text):
  try:
    match = re.match(r'([\d.]+)\s*([a-zA-Z]+)', text)
    if match:
      number = float(match.group(1))
      units = match.group(2)
      return number, units
    else:
      raise ValueError("Invalid input format")
  except Exception as e:
    return None, None

def detector(img, path_s= None, test= False):
  """
    Considerations to determine positions of SUPPORTS, LOADS and MOMENTS:
      - Loads applied on a support
        * The load box could be bigger than the support box
        * The load box could be smaller than the support box
        * The load could be overlapped but one corner is not within the support
        * The position of the load will be the same as the support
      - Distributed loads between supports
        * It requires to have the length of the distribution on the position set
      - Distributed loads from support to support
        * The start and end must fall within the supports box
        * The start and end must fall closely to the supports box, a bit less or more
        * The lenght can be deduced from the positions
      - Distributed loads passing more supports
        * Will be treated as 2 distributed loads (harmless)
        * The distances with respect to the supports are required
  """

  outputs = {
  0: "pload",
  1: "pinned",
  2: "roller",
  3: "fixed",
  4: "moment",
  5: "dload"}

  if test:
    img_np = cv2.imread(img)
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\efame\Desktop\lunwen\best.pt', force_reload=False)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\EFALTAMIRANO\Desktop\best.pt', force_reload=False)
    
    results = model(img_np)
    print(f"\nDetection #: {len(results.xyxy[0])}")
    for i, j in enumerate(results.xyxy[0]):
      print(f"Clase: {outputs[int(j[-1])]}, Prob: {j[-2] * 100}")
    
    img_rendered = results.render()[0]  # take first image
    img_rendered = cv2.cvtColor(img_rendered, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rendered)
    plt.axis("off")
    read = eso.easyocr.Reader(["es"])
    text = read.readtext(img_np)
    plt.show()

  loads_supports_detections = torch.tensor([
    [ 30.55132, 287.94385, 135.13759, 391.43811,   0.90888,   1.00000],
    [451.07086, 292.65585, 566.22668, 361.85443,   0.78141,   2.00000],
    [273.38702, 148.99152, 318.91672, 285.68210,   0.62758,   0.00000],
    [823.38702, 220.99152, 859.38702, 350.68210,   0.82758,   3.00000],
    [ 31.33456, 178.99152, 134.65932, 284.24218,   0.9758,   0.00000],])
  
  positions_detections = [
    [ 184.0, 410.0, 203.1, 460.8,   0.90888,   "4"],
    [ 367.8, 410.0, 387.8, 460.8,   0.78141,   "2m"],
    [ 682.5, 410.0, 712.5, 460.8,   0.90888,   "3.5 m"],
    [ 49.4, 142.9, 109.7, 168.3,   0.8880,   "8t"],
    [ 252.5, 130, 342.1, 142.5,   0.988,   "50 kg"],]
  
  spprts_set = list()
  dloads_set = list()
  action_set = list()
  positn_set = list()
  
  """
  Associating the elements type. Does not include the position detection, and determining the dividing line to detect loads (above) and positions (below)
  
  Missing
    * Consider error management for no positions
  """ 
  div_treshold = 0
  for i, j in enumerate(loads_supports_detections):
    if j[-1] in (1, 2, 3):
      div_treshold += j[1] / 2 + j[3] / 2
      spprts_set.append(j)
    if j[-1] in (0,4):
      action_set.append(j)
    if j[-1] == 5:
      dloads_set.append(j)

  div_treshold = div_treshold / len(spprts_set)

  "Associating the type of detection (load or position)"
  for i, j in enumerate(positions_detections):
    if (j[1] / 2 + j[3] / 2) <= div_treshold:
      action_set.append(j)
    else:
      positn_set.append(j)
      
  "Error Management"
  if len(spprts_set) == 0:
    raise NoDetections("No supports detected")
  if len(action_set) == 0 and len(dloads_set) == 0:
    raise NoDetections("No loads or moments detected")
  if len(positn_set) == 0:
    warnings.warn("\nNo positions detected. It will be proceeded to calculate positions based on the picture.", NoDetectionsWarning, stacklevel=2)
  if len(positn_set) == 0 and len(action_set) == 0 and len(spprts_set) == 0:
    return 0
  
  """
  Detect loads applied on a support
    Just record the index of the action object and when deducing
    the distances check the 'overlapped_objects'
    The overlapped objects set will be formed by the loads
  """
  
  overlapped_objects = list()

  for i, action_i in enumerate(action_set): # Loads
    ld_l, ld_r = action_i[0], action_i[2]

    # Loads applied on supports

    "Overlapped objects are important to wrapp the up and later infere their positions which will be the same to the other objects"
    for support_i in spprts_set:
      s_l, s_r = support_i[0], support_i[2]

      if s_l >= ld_r or s_r <= ld_l: # The load box is outside the support (dont c)
        continue
      elif s_l >= ld_l and s_r <= ld_r: # The load box is bigger than supports
        overlapped_objects.append(i)
      elif s_l <= ld_l and ld_r <= s_r: # The load box is smaller than supports
        overlapped_objects.append(i)
      elif ld_l < s_l and ld_r <= s_r: # The load box is is bigger on the left
        overlapped_objects.append(i)
      elif s_l <= ld_l and s_r < ld_r: # The load box is bigger on the right
        overlapped_objects.append(i)
        
  for i, dload_i in enumerate(dloads_set): # Distributed loads analysis
    d_l, d_r = dload_i[0], dload_i[2]
    for support_i in spprts_set:
      s_l, s_r = support_i[0], support_i[2]
      if d_l <= s_l and d_r >= s_r: # The distributed load passes a support
        calculate_shit = 0
        for support_i_1 in spprts_set:
          p_l_2, p_r_2 = support_i_1[0], support_i_1[2]

  for i in action_set:
    plt.plot([i[0], i[0], i[2], i[2], i[0]], [i[1], i[3], i[3], i[1], i[1]], color= "gray")
  for i in spprts_set:
    plt.plot([i[0], i[0], i[2], i[2], i[0]], [i[1], i[3], i[3], i[1], i[1]], color= "red")

  plt.plot([0, 1000], [div_treshold,div_treshold], color= "black")
  ax = plt.gca()
  ax.set_ylim(ax.get_ylim()[::-1])
  plt.show()
  return 0

class NoDetections(BaseException):
  pass

class NoDetectionsWarning(Warning):
    pass


img = r"C:\Users\efame\Desktop\photo_2023-06-16_11-39-36.jpg"
img = r"C:\Users\EFALTAMIRANO\Desktop\Untitled.jpg"
detector(img, test=False)