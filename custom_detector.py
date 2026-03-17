import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import re
import warnings

MODEL_PATH = "./best.pt"
CONF_THRESHOLD = 0.5

# YOLO classes
CLASSES = {
    0: "pload",
    1: "pinned",
    2: "roller",
    3: "fixed",
    4: "moment",
    5: "dload"
}

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=MODEL_PATH,
    force_reload=False
)

reader = easyocr.Reader(['en'])

def preprocess_image(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2)

    kernel = np.ones((2,2),np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return img, clean

def extract_number_and_units(text):

    text = text.replace(" ", "")
    match = re.search(r'(\d+\.?\d*)([a-zA-Z]*)', text)

    if match:
      number = float(match.group(1))
      units = match.group(2) if match.group(2) else None
      return number, units

    return None, None


def overlap(boxA, boxB):
  xA = max(boxA[0], boxB[0])
  xB = min(boxA[2], boxB[2])
  return max(0, xB - xA)
  
def detect_text_regions(img):
    text_data = []
    results = reader.readtext(img)
    
    for bbox, text, conf in results:
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[2][0])
        y2 = int(bbox[2][1])
        text_data.append([x1,y1,x2,y2,conf,text])

    return text_data
  
def detector(image_path, show=True):
    img, processed = preprocess_image(image_path)
    results = model(img)

    detections = results.xyxy[0].cpu().numpy()

    detections = [
        d for d in detections
        if d[4] > CONF_THRESHOLD]

    loads = []
    supports = []
    moments = []
    dloads = []

    for x1,y1,x2,y2,conf,cls in detections:
      cls = int(cls)
      obj = [x1,y1,x2,y2,conf,cls]
      if cls in (1,2,3):
        supports.append(obj)
      elif cls == 0:
        loads.append(obj)
      elif cls == 4:
        moments.append(obj)
      elif cls == 5:
        dloads.append(obj)

    if len(supports) == 0:
      raise Exception("No supports detected")

    # OCR
    text_regions = detect_text_regions(img)

    positions = []
    forces = []

    div_threshold = np.mean([(s[1]+s[3])/2 for s in supports])

    for t in text_regions:
      center = (t[1] + t[3]) / 2
      if center < div_threshold:
        forces.append(t)
      else:
        positions.append(t)

    overlapped = []
    for i, load in enumerate(loads):
      load_box = load[:4]
      for support in supports:
        support_box = support[:4]
        if overlap(load_box, support_box) > 10:
          overlapped.append(i)

    if show:
      fig, ax = plt.subplots()
      ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      for x1,y1,x2,y2,conf,cls in detections:
        color = "red"
        if cls == 0:
          color = "blue"
        if cls in (1,2,3):
          color = "green"
        if cls == 4:
          color = "purple"
        if cls == 5:
          color = "orange"

        rect = plt.Rectangle(
          (x1,y1),
          x2-x1,
          y2-y1,
          linewidth=2,
          edgecolor=color,
          facecolor='none')

        ax.add_patch(rect)

        ax.text(
          x1,
          y1-5,
          CLASSES[cls],
          color=color,
          fontsize=10)

      # text boxes
      for t in text_regions:
        x1,y1,x2,y2,conf,text = t

        rect = plt.Rectangle(
          (x1,y1),
          x2-x1,
          y2-y1,
          linewidth=1,
          edgecolor="yellow",
          facecolor='none')

        ax.add_patch(rect)
        ax.text(x1,y1-5,text,color="yellow")

      ax.axhline(div_threshold,color="white",linestyle="--")

      ax.set_ylim(ax.get_ylim()[::-1])
      plt.axis("off")
      plt.show()

    parsed_positions = []

    for p in positions:
      num, unit = extract_number_and_units(p[-1])

      parsed_positions.append({
        "value": num,
        "unit": unit,
        "bbox": p[:4]
      })

    parsed_forces = []

    for f in forces:

      num, unit = extract_number_and_units(f[-1])
      parsed_forces.append({
        "value": num,
        "unit": unit,
        "bbox": f[:4]})

    return {"supports": supports,
      "loads": loads,
      "moments": moments,
      "distributed_loads": dloads,
      "positions": parsed_positions,
      "forces": parsed_forces,
      "overlapped_loads": overlapped}

if "__main__" == __name__:
  IMAGE_PATH = "./photo_2023-02-06_13-42-10.jpg"

  result = detector(IMAGE_PATH)
  print("\nDetected Elements\n")

  for k,v in result.items():
    print(k,":",v)