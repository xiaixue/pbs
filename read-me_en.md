# PhotoBeamSolver

This application allows the analysis of beam systems by either manually defining the structural model or automatically extracting data from an image of a beam diagram.

The program computes **support reactions, shear force, bending moment, and deflection diagrams** using the `indeterminatebeam` solver. Additionally, it integrates computer vision (local model or GPT-based) to interpret beam diagrams directly from images.

---

## Overview

You can use the application in two ways:

### 1. Manual Input
Define:
- Beam properties (E, A, I, L)
- Supports
- Loads

### 2. Image-Based Input
- Load a beam diagram (`.png`, `.jpg`, `.jpeg`)
- Choose detection mode:
  - **Local model**
  - **GPT Vision**
- The system extracts beam data automatically

---

## Features

- Interactive GUI built with `tkinter` + `customtkinter`
- Beam analysis using `indeterminatebeam`
- Automatic diagram detection from images
- Multiple load types:
  - Point loads
  - Distributed loads
  - Moments
- Real-time visualization:
  - Beam diagram
  - Shear force diagram
  - Bending moment diagram
  - Deflection curve

---

## Detection Modes

### Local Detection
Uses a custom model:
```python
custom_detector.detector(image_path)
```

### GPT Vision Detection
Uses the OpenAI API to parse beam diagrams:

- Extracts:
  - Beam length
  - Supports
  - Loads
- Returns structured JSON

> Requires `OPENAI_API_KEY` environment variable.

---

## Data Format

### Internal Solver Format

```python
{
  "Length": float,
  "supports": [
    ("pinned" | "roller" | "fixed", position)
  ],
  "loads": [
    ("pload", magnitude, position),
    ("dload", mag_start, x_start, mag_end, x_end),
    ("moment", magnitude, position)
  ]
}
```

---

## Core Components

### `BeamDetector`

Handles detection mode.

```python
detect(image_path)
```

- `"local"` → uses custom model  
- `"gpt"` → uses GPT Vision  

---

### `BeamSettings`

Main interface for:
- Editing beam properties
- Managing supports and loads
- Running analysis

---

### `solve_Beam(...)`

Performs structural analysis.

**Returns:**
- reactions
- shear force
- bending moment
- deflection

---

### `plotter(...)`

Generates diagrams:
- Beam + reactions
- Moment diagram
- Shear diagram
- Deflection curve

---

### `parse_beam_image(...)`

Processes image and sends it to GPT:

- Preprocessing with `cv2`
- Encodes image to base64
- Sends structured prompt
- Returns JSON

---

## Image Preprocessing

```python
preprocess_beam_image(...)
```

Steps:
- Grayscale conversion
- Adaptive thresholding
- Noise reduction
- Resize to 512×512

---

## Units Handling

The application supports:
- Metric (cm, kg)
- Imperial (in, lb)

Automatic conversion is applied before solving.

---

## GUI Structure

### `Home`
- Entry point
- Select:
  - Manual input
  - Image input

### `Load_Picture`
- Upload image
- Select detection mode

### `BeamSettings`
- Configure system
- View results in tabs

---

## Tabs

- **Beam** → supports & reactions  
- **Moment** → bending moment diagram  
- **Shearing** → shear force diagram  
- **Deflection** → displacement curve  

---

## Example Workflow

Run `gui_PBS.py`

1. Choose **Load a picture**
2. Select a beam diagram
3. Review extracted data
4. Click **Continue**
5. View diagrams

---

## Requirements

```bash
pip install opencv-python pillow torch numpy matplotlib customtkinter indeterminatebeam openai
```

---

## Notes

- Image detection accuracy depends on diagram quality
- GPT mode may estimate values if unclear
- Large models or many loads may increase computation time
- Local detector must be properly configured (`custom_detector`)