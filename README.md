# Fabry–Perot Interferometer Simulator

An interactive Python teaching tool for exploring how multiple reflections build up in a Fabry–Perot cavity, how the reflected waves cancel at resonance, and how tilted illumination produces circular fringes. The interface starts in English. Use the buttons at the top of any window to switch to French or Chinese.

[中文使用说明](#中文使用说明)
<img width="1257" height="746" alt="image" src="https://github.com/user-attachments/assets/35c0d746-a433-413e-aa7c-0dd45a6084d0" />

## Run locally

Use a desktop environment with a graphical Matplotlib backend such as TkAgg or QtAgg. Python 3.12 is the version used for testing. Open a terminal in the folder containing [`fabry_perot.py`](fabry_perot.py) and choose one of the following methods.

### Python with `venv` and `pip`

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib
.\.venv\Scripts\python.exe .\fabry_perot.py
```

If Windows provides the `py` launcher, you can use `py -3.12 -m venv .venv` for the first line.

**macOS or Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
.venv/bin/python fabry_perot.py
```

If NumPy and Matplotlib are already installed in your Python environment, run `python fabry_perot.py` or `python3 fabry_perot.py` with that interpreter.

### Conda

```bash
conda create --name fabry-perot python=3.12 numpy matplotlib
conda run --name fabry-perot python fabry_perot.py
```

### uv

```bash
uv run --no-project --python 3.12 --with numpy --with matplotlib fabry_perot.py
```

If Matplotlib starts with a non-interactive backend such as Agg, use a desktop Python installation with Tk or Qt bindings. If Chinese characters appear as boxes, install a system font such as Noto Sans CJK SC.

## Three experiments

| Window | What to try |
| --- | --- |
| **Fabry–Perot interferometer** | Press **Play** to watch the wavefronts propagate, or **Step** to advance one frame. At zero detuning, **Jump to steady state** shows the cavity leakage cancelling the direct reflection while the incident wave keeps moving. |
<img width="1254" height="745" alt="image" src="https://github.com/user-attachments/assets/ac2a40e8-d0da-4a45-a2bf-aba02dbbdbf2" />

| Window | What to try |
| --- | --- |
| **Angle and beam width** | Set the reflectance to `0.99` and the angle to `25°`. With the center angle locked to resonance, compare a wide beam at `w/e = 1000` with a narrow beam at `w/e = 0.1`. The center angle remains resonant while total transmission drops from about 99.16% to 0.50%. |
<img width="1254" height="748" alt="image" src="https://github.com/user-attachments/assets/4c774608-fe47-4dca-a485-0153cecfb489" />

| Window | What to try |
| --- | --- |
| **Transmission and rings** | Compare Airy peaks and ring patterns under Gaussian and uniform illumination. Click a point in a ring image to follow its angle through the diverging lens, the cavity mirrors, and the focusing lens to the focal-plane screen. |
<img width="1258" height="744" alt="image" src="https://github.com/user-attachments/assets/df4d325b-fc58-48dc-8384-021cf4818de7" />


## Model

The cavity uses identical, parallel, lossless mirrors. The time-domain animation shows scalar waves at normal incidence, including the travel time for each round trip. The finite-beam window uses a steady-state Gaussian angular spectrum. The ring window uses an axisymmetric illumination model and ideal thin lenses.

## 中文使用说明

这是一个 Fabry–Perot 干涉仪教学程序，可以观察光源开启后的电场、相量和功率，比较斜入射时宽束与窄束的透射，并查看干涉圆环和从光源到焦平面的完整光路。

在 `fabry_perot.py` 所在目录运行上面的任一组命令。普通 Python 用户可用 [venv 和 pip](#python-with-venv-and-pip)，也可以使用 [Conda](#conda) 或 [uv](#uv)。窗口打开后，点右上角的 **中文** 切换界面，再点 **播放** 开始观察。
