# Running on macOS

The code is pure Python with no platform-specific calls, so it runs on macOS
unchanged — Intel and Apple Silicon alike. The only real macOS gotcha is
**tkinter**, which some Python installs ship without. That affects `gui.py`
only; the command-line pipeline works regardless.

---

## 1. Pick a Python that has tkinter

This is the step that trips people up. Check first:

```bash
python3 -c "import tkinter; print('tkinter OK, Tk', tkinter.TkVersion)"
```

You want **Tk 8.6 or newer**. Depending on what you see:

| Result | What to do |
|---|---|
| `Tk 8.6` or higher | You're set — go to step 2. |
| `ModuleNotFoundError: No module named 'tkinter'` | Homebrew Python omits it: `brew install python-tk` |
| `Tk 8.5` | Too old — CustomTkinter renders badly on 8.5. Install Python from [python.org](https://www.python.org/downloads/macos/), which bundles Tk 8.6. |
| Using **pyenv** | pyenv builds Python without Tk unless it was present at build time: `brew install tcl-tk` then reinstall your Python version. |

The simplest reliable option is the **python.org installer**, which includes a
working Tk 8.6 out of the box. Requires Python 3.9 or newer (3.11+ recommended;
the code uses `X | Y` type syntax).

## 2. Create the virtual environment

From the project folder:

```bash
cd "AUTO4507 - RobMan"
python3 -m venv .venv
source .venv/bin/activate
```

Note macOS uses `source .venv/bin/activate`, not the Windows
`.venv\Scripts\Activate.ps1`. Your prompt should now start with `(.venv)`.

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Both `numpy` and `opencv-python` publish native **arm64** wheels, so this is a
fast binary install on Apple Silicon — no Rosetta, no compiling.

> If `opencv-python` fails to build or resolve, use
> `pip install opencv-python-headless` instead. This project never calls
> `cv2.imshow` (all display goes through tkinter), so the headless build works
> perfectly here.

## 4. Run it

```bash
# Verify everything works - 13 checks
python verify.py

# Command line: print each shape and its centre point
python pipeline.py test.webp

# The stage-by-stage GUI
python gui.py
```

`python gui.py` opens the viewer on `test.webp`; pass another path to start
elsewhere, or use the **Load image…** button. On first launch macOS may take a
second to bring the window to the front — click the Python icon in the Dock if
it opens behind your terminal.

Expected output from `python pipeline.py test.webp`:

```
type              centre   conf  verts   solid   circ  aspect
-------------------------------------------------------------
square        (409, 394)   0.99      4   0.995  0.854    1.00
semicircle    (413, 178)   0.99      6   0.974  0.718    1.63
triangle      (162, 185)   0.90      4   0.965  0.662    1.08
star          (169, 402)   0.99      5   0.879  0.678    1.04
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tkinter'`**
Homebrew Python without Tk. Run `brew install python-tk`, or switch to the
python.org build. Re-create the venv afterwards so it picks up the new Python.

**The GUI opens but widgets look wrong / oversized**
Almost always Tk 8.5. Confirm with the check in step 1 and move to a Tk 8.6 build.

**`python: command not found`**
Use `python3` before activating the venv. After `source .venv/bin/activate`,
plain `python` refers to the venv's interpreter.

**`cv2.error` about `convexityDefects`**
The code already normalises the differing return shapes between OpenCV 4
(`(N,1,4)`) and OpenCV 5 (`(N,4)`), so either major version is fine. If you see
this, report which `cv2.__version__` you have.

**Window opens behind the terminal**
Normal macOS behaviour for Tk apps. Click the Dock icon or use ⌘-Tab.

---

## Note for the UR5 lab

The pipeline reports centres in **pixel coordinates**, with the origin at the
image's top-left and `y` increasing downward. Before the arm can pick anything,
those need converting to robot coordinates, which requires two things this
project does not do:

1. **Camera calibration** — `cv2.calibrateCamera` on a checkerboard, to correct
   lens distortion and recover the intrinsics.
2. **Hand–eye calibration** — `cv2.calibrateHandEye`, to get the transform
   between the camera frame and the robot base frame.

With the board lying flat at a known, fixed height, the shortcut is a
**homography**: capture four points whose robot XY coordinates you know, pass
them with their pixel coordinates to `cv2.getPerspectiveTransform`, then map any
detected centre through it with `cv2.perspectiveTransform`. That is usually
accurate enough for flat pick-and-place and avoids full 3D calibration.

`shape.centre` in `pipeline.py` is the value to feed into that transform. The
centres are stable to within about 5 px across every lighting variation tested,
so the segmentation is unlikely to be the limiting factor in pick accuracy —
calibration will be.
