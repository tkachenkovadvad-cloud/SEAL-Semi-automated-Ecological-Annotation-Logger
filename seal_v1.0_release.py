# -*- coding: utf-8 -*-
"""
SEAL — Semi-automated Ecological Annotation Logger (3-tab GUI)

Designed for:
- Seals (individual tracking / point marking)
- Penguins and penguin nests (high-density point marking + event scoring)
- Any other repeated observations in time-lapse / survey imagery

==================================================
WORKFLOW
==================================================

Tab 1 (Folder/Time):
- Select a folder with image frames (JPG/PNG/TIFF/BMP). Files are sorted naturally.
- Choose ONE time-mapping mode:

  A) MARK & TRACK (continuous survey / motion expected)
  - Enter start/end timestamps for the FIRST and LAST photo
  - Enter step (minutes) and pick a "Start frame"
  - Apply -> enables Tab 2 and Tab 3
  - Enables optional template-based tracking in Tab 3

  B) MARK & VIEW (observation / no continuity assumed; recommended for CEMP / penguins)
  - Enter dataset start/end datetime
  - Enter daily timeframe start/end (HH:MM:SS)
  - Enter step (minutes) and pick a "Start frame"
  - Apply -> enables Tab 2 and Tab 3
  - Tracking is disabled; all marking is manual

Tab 2 (Objects & Events):
- Add any number of Object IDs:
  - Seals: individual IDs (Seal_01, Seal_02, ...)
  - Penguins / nests: IDs can represent nests, territories, or groups (Nest_001, Nest_002, ...)
- Add any number of Event Classes (name + hotkey)
- For each class add any number of Events (name + hotkey)
- You can return here anytime and add more (session-based)

Tab 3 (Play):
- Play forward / Play back through frames as a slideshow.
- Space = pause/resume (keeps last direction).

While paused:
- Left click to place a marker:
  - If clicking near an existing marker -> drag to adjust
  - Otherwise -> popup asks which Object ID to place
- Log event via hotkeys:
  - Press Class hotkey, then Event hotkey (within that class)
- Add comment: C
- Toggle object ACTIVE/INACTIVE: X
- Zoom:
  - Mouse wheel = zoom (cursor-centered)
  - Z = reset zoom
  - R = reset view (fit)

Penguin-friendly notes:
- MARK & VIEW is recommended for penguin colonies (frames may be sparse and time gaps irregular).
- Use Events to score nest state / attendance / behavior (e.g., INCUBATING, BROODING, FEEDING, ABSENT).
- Use multiple Object IDs for fixed nests/territories, or use group IDs if individual identification is not needed.

==================================================
TRACKING (MARK & TRACK MODE ONLY)
==================================================

- Simple per-object template matching (grayscale) with local search window.
- Auto-pauses if any tracked object confidence < min_conf.
- Tracking parameters:
  - min_conf (0.0–1.0): minimum confidence threshold
  - search_radius (pixels): local search window radius

==================================================
OUTPUT
==================================================

- Save CSV writes:
  - tracks.csv  (marker positions + tracking detections/adjustments)
  - events.csv  (events + comments)
  to the selected image folder.

==================================================
DEPENDENCIES
==================================================

pip install opencv-python pillow numpy
"""

from __future__ import annotations

import os
import sys
import re
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import copy


# --------------------------- utils ---------------------------

def natural_key(s: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_dt(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")

def parse_time_hms(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    # Accept H:MM:SS or HH:MM:SS
    parts = s.split(':')
    if len(parts) != 3:
        raise ValueError('Time must be HH:MM:SS')
    h, m, sec = (int(parts[0]), int(parts[1]), int(parts[2]))
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        raise ValueError('Time values out of range')
    return h, m, sec

def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def pil_bilinear():
    """
    Pillow compatibility:
    - Pillow < 10: Image.BILINEAR exists
    - Pillow >= 10: constants moved to Image.Resampling
    """
    resampling = getattr(Image, "Resampling", Image)
    return resampling.BILINEAR


# --------------------------- models ---------------------------

@dataclass
class TrackState:
    x: Optional[int] = None
    y: Optional[int] = None
    template: Optional[np.ndarray] = None
    conf: float = 0.0
    status: str = "unset"  # unset / tracked / lost / inactive

@dataclass
class EventClass:
    name: str
    hotkey: str  

@dataclass
class EventDef:
    name: str
    hotkey: str  

@dataclass
class EventEntry:
    frame_idx: int
    t_real: str
    file_name: str
    object_id: str
    class_name: str
    event_name: str
    note: str = ""

@dataclass
class TrackLogEntry:
    frame_idx: int
    t_real: str
    file_name: str
    object_id: str
    x: int
    y: int
    conf: float
    status: str
    source: str  


# --------------------------- app ---------------------------

class SealGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SEAL v1.0")
        self.geometry("1400x860")

        def _resource_path(rel: str) -> str:
            base = getattr(
                sys,
                "_MEIPASS",
                os.path.dirname(os.path.abspath(__file__))
            )
            return os.path.join(base, rel)

        try:
            self.iconbitmap(_resource_path("seal.ico"))
        except Exception:
            pass
        
        self.folder: Optional[str] = None
        self.files: List[str] = []       
        self.file_names: List[str] = []  
        self.idx: int = 0
        self.start_idx: int = 0

        self.t_start: Optional[datetime] = None
        self.t_end: Optional[datetime] = None
        self.step_min: float = 10.0

        self.ts_mode: str = "track"
        self.t_map: Optional[List[datetime]] = None

        self.camera_name: str = ""

        self.object_ids: List[str] = []
        self.track: Dict[str, TrackState] = {}

        self.event_classes: Dict[str, EventClass] = {}
        self.class_events: Dict[str, Dict[str, EventDef]] = {}
        self.pending_class: Optional[str] = None
        self._grave_down: bool = False

        self.events_log: List[EventEntry] = []

        self.markers_by_frame: Dict[int, Dict[str, Tuple[int, int, str, float]]] = {}
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []
        self._drag_undo_pushed: bool = False

        self.track_log: List[TrackLogEntry] = []

        self.search_radius: int = 140
        self.tpl_half: int = 26
        self.min_conf: float = 0.55
        self.match_method = cv2.TM_CCOEFF_NORMED

        self.playing: bool = False
        self.play_dir: int = +1  # +1 forward, -1 back
        self.seconds_per_frame: float = 0.20
        self.after_id: Optional[str] = None

        self._tkimg: Optional[ImageTk.PhotoImage] = None
        self._disp_scale: float = 1.0
        self._disp_offx: int = 0
        self._disp_offy: int = 0
        self.zoom: float = 1.0
        self.zoom_min: float = 0.25
        self.zoom_max: float = 6.0
        self._view_offx: Optional[int] = None
        self._view_offy: Optional[int] = None
        self._zoom_anchor: Optional[Tuple[float, float, int, int]] = None  # (ix,iy,cx,cy)
        self._current_bgr_cache: Optional[np.ndarray] = None
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_max: int = 8

        self._dragging_id: Optional[str] = None

        self._build_ui()
        self._bind_keys()
        self._update_tabs_enabled()
        self._render_placeholder()

    # --------------------------- UI ---------------------------

    def _build_ui(self) -> None:
        # --- Main vertical splitter: top=UI, bottom=Console ---
        self.pw_main = ttk.Panedwindow(self, orient="vertical")
        self.pw_main.pack(fill="both", expand=True)

        self.top_area = ttk.Frame(self.pw_main)
        self.bottom_area = ttk.Frame(self.pw_main)
        self.pw_main.add(self.top_area, weight=5)
        self.pw_main.add(self.bottom_area, weight=1)

        self.nb = ttk.Notebook(self.top_area)
        self.nb.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        self.tab1 = ttk.Frame(self.nb)
        self.tab2 = ttk.Frame(self.nb)
        self.tab3 = ttk.Frame(self.nb)

        self.nb.add(self.tab1, text="1) Folder & Time")
        self.nb.add(self.tab2, text="2) Objects & Events")
        self.nb.add(self.tab3, text="3) Play")

        console_frame = ttk.Frame(self.bottom_area)
        console_frame.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Label(console_frame, text="Console").pack(side="top", anchor="w")
        self.txt = tk.Text(console_frame, height=10, wrap="none")
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(console_frame, orient="vertical", command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

        t1 = ttk.Frame(self.tab1)
        t1.pack(fill="both", expand=True, padx=10, pady=10)

        row0 = ttk.Frame(t1); row0.pack(fill="x")
        ttk.Button(row0, text="Select folder", command=self.ui_select_folder).pack(side="left")
        self.lbl_folder = ttk.Label(row0, text="(no folder)")
        self.lbl_folder.pack(side="left", padx=10)

        lf_track = ttk.LabelFrame(t1, text="images to mark&track")
        lf_track.pack(fill="x", pady=(12, 0))

        r1 = ttk.Frame(lf_track); r1.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(r1, text="Start timestamp (first photo):").pack(side="left")
        self.ent_tstart = ttk.Entry(r1, width=20)
        self.ent_tstart.insert(0, "2026-01-01 00:00:00")
        self.ent_tstart.pack(side="left", padx=6)

        ttk.Label(r1, text="End timestamp (last photo):").pack(side="left", padx=(12, 0))
        self.ent_tend = ttk.Entry(r1, width=20)
        self.ent_tend.insert(0, "2026-01-01 23:59:59")
        self.ent_tend.pack(side="left", padx=6)

        r2 = ttk.Frame(lf_track); r2.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(r2, text="Step (minutes):").pack(side="left")
        self.ent_step = ttk.Entry(r2, width=8)
        self.ent_step.insert(0, "10")
        self.ent_step.pack(side="left", padx=6)

        ttk.Label(r2, text="Start frame:").pack(side="left", padx=(12, 0))
        self.cmb_startframe_track = ttk.Combobox(r2, width=55, state="readonly", values=[])
        self.cmb_startframe_track.pack(side="left", padx=6)

        ttk.Button(r2, text="Apply", command=self.ui_apply_mark_track).pack(side="left", padx=10)

        r3 = ttk.Frame(lf_track); r3.pack(fill="x", padx=8, pady=(6, 8))
        ttk.Label(r3, text="Camera:").pack(side="left")
        self.ent_camera_track = ttk.Entry(r3, width=24)
        self.ent_camera_track.insert(0, "")
        self.ent_camera_track.pack(side="left", padx=6)

        lf_view = ttk.LabelFrame(t1, text="images to mark&view")
        lf_view.pack(fill="x", pady=(12, 0))

        v1 = ttk.Frame(lf_view); v1.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(v1, text="Dataset start (YYYY-MM-DD HH:MM:SS):").pack(side="left")
        self.ent_ds_start = ttk.Entry(v1, width=20)
        self.ent_ds_start.insert(0, "2026-01-01 00:00:00")
        self.ent_ds_start.pack(side="left", padx=6)

        ttk.Label(v1, text="Dataset end (YYYY-MM-DD HH:MM:SS):").pack(side="left", padx=(12, 0))
        self.ent_ds_end = ttk.Entry(v1, width=20)
        self.ent_ds_end.insert(0, "2026-01-01 23:59:59")
        self.ent_ds_end.pack(side="left", padx=6)

        v2 = ttk.Frame(lf_view); v2.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(v2, text="Timeframe start (HH:MM:SS):").pack(side="left")
        self.ent_tf_start = ttk.Entry(v2, width=10)
        self.ent_tf_start.insert(0, "08:30:00")
        self.ent_tf_start.pack(side="left", padx=6)

        ttk.Label(v2, text="Timeframe end (HH:MM:SS):").pack(side="left", padx=(12, 0))
        self.ent_tf_end = ttk.Entry(v2, width=10)
        self.ent_tf_end.insert(0, "14:30:00")
        self.ent_tf_end.pack(side="left", padx=6)

        ttk.Label(v2, text="Step (minutes):").pack(side="left", padx=(12, 0))
        self.ent_step_view = ttk.Entry(v2, width=8)
        self.ent_step_view.insert(0, "30")
        self.ent_step_view.pack(side="left", padx=6)

        v3 = ttk.Frame(lf_view); v3.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(v3, text="Start frame:").pack(side="left")
        self.cmb_startframe_view = ttk.Combobox(v3, width=55, state="readonly", values=[])
        self.cmb_startframe_view.pack(side="left", padx=6)

        ttk.Button(v3, text="Apply", command=self.ui_apply_mark_view).pack(side="left", padx=10)

        v4 = ttk.Frame(lf_view); v4.pack(fill="x", padx=8, pady=(6, 8))
        ttk.Label(v4, text="Camera:").pack(side="left")
        self.ent_camera_view = ttk.Entry(v4, width=24)
        self.ent_camera_view.insert(0, "")
        self.ent_camera_view.pack(side="left", padx=6)

        self.lbl_tab1_info = ttk.Label(t1, text="Load folder to populate frames.")
        self.lbl_tab1_info.pack(anchor="w", pady=(10, 0))

        t2 = ttk.Frame(self.tab2)
        t2.pack(fill="both", expand=True, padx=10, pady=10)

        preset_row = ttk.Frame(t2)
        preset_row.pack(fill="x", pady=(0, 10))
        ttk.Button(preset_row, text="Load preset (.txt)", command=self.ui_load_preset).pack(side="left")
        ttk.Label(preset_row, text="Preset loads animal IDs + event classes + events").pack(side="left", padx=10)

        left = ttk.LabelFrame(t2, text="Objects (Animal IDs)")
        left.pack(side="left", fill="both", expand=False, padx=(0, 10))

        addrow = ttk.Frame(left); addrow.pack(fill="x", pady=6, padx=6)
        ttk.Label(addrow, text="New ID:").pack(side="left")
        self.ent_new_animal = ttk.Entry(addrow, width=14)
        self.ent_new_animal.pack(side="left", padx=6)
        ttk.Button(addrow, text="+", width=3, command=self.ui_add_animal).pack(side="left")
        ttk.Button(addrow, text="Remove", command=self.ui_remove_animal).pack(side="left", padx=6)

        self.lst_animals = tk.Listbox(left, height=18, exportselection=False)
        self.lst_animals.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.lst_animals.bind("<<ListboxSelect>>", lambda _e: self._sync_play_animal_dropdown())

        mid = ttk.LabelFrame(t2, text="Event Classes")
        mid.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cadd = ttk.Frame(mid); cadd.pack(fill="x", pady=6, padx=6)
        ttk.Label(cadd, text="Class name:").pack(side="left")
        self.ent_class_name = ttk.Entry(cadd, width=14)
        self.ent_class_name.pack(side="left", padx=6)
        ttk.Label(cadd, text="Hotkey:").pack(side="left")
        self.ent_class_key = ttk.Entry(cadd, width=4)
        self.ent_class_key.pack(side="left", padx=6)
        ttk.Button(cadd, text="+", width=3, command=self.ui_add_class).pack(side="left")
        ttk.Button(cadd, text="Remove", command=self.ui_remove_class).pack(side="left", padx=6)

        self.lst_classes = tk.Listbox(mid, height=18, exportselection=False)
        self.lst_classes.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.lst_classes.bind("<<ListboxSelect>>", lambda _e: self.ui_refresh_events_list())

        right = ttk.LabelFrame(t2, text="Events (for selected class)")
        right.pack(side="left", fill="both", expand=True)

        eadd = ttk.Frame(right); eadd.pack(fill="x", pady=6, padx=6)
        ttk.Label(eadd, text="Event name:").pack(side="left")
        self.ent_event_name = ttk.Entry(eadd, width=16)
        self.ent_event_name.pack(side="left", padx=6)
        ttk.Label(eadd, text="Hotkey:").pack(side="left")
        self.ent_event_key = ttk.Entry(eadd, width=4)
        self.ent_event_key.pack(side="left", padx=6)
        ttk.Button(eadd, text="+", width=3, command=self.ui_add_event).pack(side="left")
        ttk.Button(eadd, text="Remove", command=self.ui_remove_event).pack(side="left", padx=6)

        self.lst_events = tk.Listbox(right, height=18, exportselection=False)
        self.lst_events.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        info = ttk.Label(t2, text="Hotkeys: while PAUSED in Play tab, press Class hotkey, then Event hotkey to log.")
        info.pack(side="bottom", anchor="w", pady=(10, 0))

        t3 = ttk.Frame(self.tab3)
        t3.pack(fill="both", expand=True, padx=10, pady=10)

        top = ttk.Frame(t3); top.pack(fill="x")

        ttk.Button(top, text="Play forward", command=lambda: self.ui_start_play(+1)).pack(side="left")
        ttk.Button(top, text="Play back", command=lambda: self.ui_start_play(-1)).pack(side="left", padx=6)
        ttk.Button(top, text="<-", width=4, command=lambda: self._step_once(-1)).pack(side="left", padx=6)
        ttk.Button(top, text="->", width=4, command=lambda: self._step_once(+1)).pack(side="left")
        ttk.Button(top, text="Save CSV", command=self.ui_save_csv).pack(side="left", padx=16)

        ttk.Label(top, text="Speed (sec/frame):").pack(side="left", padx=(20, 0))
        self.ent_speed = ttk.Entry(top, width=8)
        self.ent_speed.insert(0, "0.20")
        self.ent_speed.pack(side="left", padx=6)

        ttk.Label(top, text="Selected animal for events:").pack(side="left", padx=(20, 0))
        self.cmb_active_animal = ttk.Combobox(top, width=14, state="readonly", values=[])
        self.cmb_active_animal.pack(side="left", padx=6)

        ttk.Label(top, text="min_conf:").pack(side="left", padx=(20, 0))
        self.ent_minconf = ttk.Entry(top, width=6)
        self.ent_minconf.insert(0, str(self.min_conf))
        self.ent_minconf.pack(side="left", padx=6)

        ttk.Label(top, text="search_r:").pack(side="left")
        self.ent_searchr = ttk.Entry(top, width=6)
        self.ent_searchr.insert(0, str(self.search_radius))
        self.ent_searchr.pack(side="left", padx=6)

        ttk.Button(top, text="Apply track params", command=self.ui_apply_track_params).pack(side="left", padx=10)

        self.lbl_play_status = ttk.Label(t3, text="(load folder in tab 1)")
        self.lbl_play_status.pack(anchor="w", pady=(8, 6))

        self.canvas = tk.Canvas(t3, bg="black", width=1200, height=680, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)      # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)  # Linux scroll down

    def _bind_keys(self) -> None:
        # Undo/Redo
        self.bind_all("<Control-z>", self.ui_undo)
        self.bind_all("<Control-Z>", self.ui_undo)
        self.bind_all("<Control-y>", self.ui_redo)
        self.bind_all("<Control-Y>", self.ui_redo)

        self.bind("<space>", self.on_space)
        self.bind("<Escape>", lambda _e: self._clear_pending_class())
        self.bind("x", lambda _e: self.toggle_active_current_animal())
        self.bind("X", lambda _e: self.toggle_active_current_animal())
        self.bind("c", lambda _e: self.ui_add_comment())
        self.bind("C", lambda _e: self.ui_add_comment())
        self.bind("z", lambda _e: self.set_zoom(1.0))
        self.bind("Z", lambda _e: self.set_zoom(1.0))
        self.bind("<Left>", lambda _e: self._step_once(-1))
        self.bind("<Right>", lambda _e: self._step_once(+1))
        self.bind_all("<KeyPress>", self.on_key_any)
        self.bind_all("<KeyRelease>", self.on_key_release_any)

        def _hk(bank: int, digit: int) -> None:
            n = 10 if digit == 0 else digit
            slot = bank + n
            if bank == 0 and getattr(self, "_grave_down", False):
                slot = 20 + n
            self._select_animal_slot(slot)

        for d in range(10):
            k = str(d)
            self.bind_all(f"<Control-KeyPress-{k}>", lambda e, dd=d: _hk(0, dd))
            self.bind_all(f"<Control-Shift-KeyPress-{k}>", lambda e, dd=d: _hk(10, dd))
            self.bind_all(f"<Control-Alt-KeyPress-{k}>", lambda e, dd=d: _hk(20, dd))

        self.bind_all("<Control-KeyPress-grave>", lambda e: setattr(self, "_grave_down", True))
        self.bind_all("<KeyRelease-grave>", lambda e: setattr(self, "_grave_down", False))

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt.insert("end", f"[{ts}] {msg}\n")
        self.txt.see("end")

    def _update_tabs_enabled(self) -> None:
        ready = bool(self.files) and (self.t_start is not None) and (self.step_min > 0)
        self.nb.tab(self.tab2, state=("normal" if ready else "disabled"))
        self.nb.tab(self.tab3, state=("normal" if ready else "disabled"))

    # --------------------------- Tab 1 ---------------------------

    def ui_select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder with frames (JPG/PNG)")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        names = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        names.sort(key=natural_key)
        if not names:
            messagebox.showerror("Error", "No image files found in this folder.")
            return

        self.folder = folder
        self.file_names = names
        self.files = [os.path.join(folder, n) for n in names]
        self.lbl_folder.config(text=folder)

        self.cmb_startframe_track.configure(values=self.file_names)
        self.cmb_startframe_view.configure(values=self.file_names)
        self.cmb_startframe_track.current(0)
        self.cmb_startframe_view.current(0)
        self.start_idx = 0
        self.idx = 0

        self._cache.clear()
        self._current_bgr_cache = None

        self.lbl_tab1_info.config(text=f"Frames: {len(self.files)} (sorted). Choose Start frame and Apply.")
        self.log(f"Folder loaded: {folder} ({len(self.files)} frames)")

        self._update_tabs_enabled()
        self._render_placeholder()


    def _set_start_frame_from_combobox(self, cmb: ttk.Combobox) -> None:
        sel = cmb.get().strip()
        self.start_idx = self.file_names.index(sel) if sel in self.file_names else 0
        self.idx = self.start_idx

    def ui_apply_mark_track(self) -> None:
        """Apply linear time mapping (first photo datetime + step*idx)."""
        if not self.files:
            messagebox.showerror("Error", "Select folder first.")
            return
        try:
            t_start = parse_dt(self.ent_tstart.get())
            t_end = parse_dt(self.ent_tend.get())
            step_min = float(self.ent_step.get())
            if step_min <= 0:
                raise ValueError("Step must be > 0")
        except Exception as e:
            messagebox.showerror("Error", f"Bad time parameters: {e}")
            return

        self.ts_mode = "track"
        self.t_start = t_start
        self.t_end = t_end
        self.step_min = step_min
        self.camera_name = self.ent_camera_track.get().strip()

        self.t_map = [self.t_start + timedelta(minutes=self.step_min * i) for i in range(len(self.files))]

        if self.t_map:
            last = self.t_map[-1]
            if self.t_end is not None and last > self.t_end + timedelta(seconds=1):
                self.log("WARN: computed last timestamp exceeds 'End timestamp'. Check step/end inputs.")

        self._set_start_frame_from_combobox(self.cmb_startframe_track)

        self.playing = False
        self.after_cancel_safe()

        self.log(
            f"Apply (track): start={fmt_dt(self.t_start)}, end={fmt_dt(self.t_end)}, "
            f"step={self.step_min}min, start_frame={self.file_names[self.start_idx]}, camera='{self.camera_name}'"
        )
        self._update_tabs_enabled()

        self.nb.select(self.tab2)
        self._sync_play_animal_dropdown()
        self._refresh_play_status()
        self._restore_frame(self.idx)
        self._show_frame(self.idx)
        self._snapshot_frame(self.idx)

    def ui_apply_mark_view(self) -> None:

        if not self.files:
            messagebox.showerror("Error", "Select folder first.")
            return
        try:
            ds_start = parse_dt(self.ent_ds_start.get())
            ds_end = parse_dt(self.ent_ds_end.get())
            if ds_end < ds_start:
                raise ValueError("Dataset end must be >= dataset start")
            h1, m1, s1 = parse_time_hms(self.ent_tf_start.get())
            h2, m2, s2 = parse_time_hms(self.ent_tf_end.get())
            step_min = float(self.ent_step_view.get())
            if step_min <= 0:
                raise ValueError("Step must be > 0")
        except Exception as e:
            messagebox.showerror("Error", f"Bad CEMP parameters: {e}")
            return

        tf_start_sec = h1 * 3600 + m1 * 60 + s1
        tf_end_sec = h2 * 3600 + m2 * 60 + s2
        if tf_end_sec < tf_start_sec:
            messagebox.showerror("Error", "Timeframe end must be >= timeframe start (same day).")
            return

        grid: List[datetime] = []
        day = ds_start.date()
        last_day = ds_end.date()
        step_td = timedelta(minutes=step_min)

        while day <= last_day:
            t0 = datetime(day.year, day.month, day.day, h1, m1, s1)
            t1 = datetime(day.year, day.month, day.day, h2, m2, s2)
            t = t0
            while t <= t1:
                if ds_start <= t <= ds_end:
                    grid.append(t)
                t += step_td
            day = day + timedelta(days=1)

        if not grid:
            messagebox.showerror("Error", "No timestamps generated. Check dataset range + timeframe.")
            return

        self.ts_mode = "view"
        self.t_start = ds_start
        self.t_end = ds_end
        self.step_min = step_min
        self.camera_name = self.ent_camera_view.get().strip()

        if len(grid) < len(self.files):
            self.log(f"WARN: generated grid has {len(grid)} timestamps but folder has {len(self.files)} frames. Extra frames will keep last timestamp.")
            grid = grid + [grid[-1]] * (len(self.files) - len(grid))
        self.t_map = grid[:len(self.files)]

        self._set_start_frame_from_combobox(self.cmb_startframe_view)

        self.playing = False
        self.after_cancel_safe()

        self.log(
            f"Apply (view): dataset={fmt_dt(ds_start)}..{fmt_dt(ds_end)}, "
            f"timeframe={self.ent_tf_start.get().strip()}..{self.ent_tf_end.get().strip()}, step={step_min}min, "
            f"start_frame={self.file_names[self.start_idx]}, camera='{self.camera_name}'"
        )
        self._update_tabs_enabled()

        self.nb.select(self.tab2)
        self._sync_play_animal_dropdown()
        self._refresh_play_status()
        self._restore_frame(self.idx)
        self._show_frame(self.idx)
        self._snapshot_frame(self.idx)

    def ui_apply_tab1(self) -> None:
        self.ui_apply_mark_track()


    # --------------------------- Tab 2 ---------------------------

    def ui_add_animal(self) -> None:
        aid = self.ent_new_animal.get().strip()
        if not aid:
            return
        if aid in self.object_ids:
            messagebox.showerror("Error", "Animal ID already exists.")
            return
        self.object_ids.append(aid)
        self.track[aid] = TrackState()
        self._refresh_animals_list()
        self.ent_new_animal.delete(0, "end")
        self.log(f"Added animal: {aid}")
        self._sync_play_animal_dropdown()
        self._show_frame(self.idx)

    def ui_remove_animal(self) -> None:
        sel = self._get_selected_listbox_value(self.lst_animals)
        if not sel:
            return
        if messagebox.askyesno("Remove", f"Remove animal '{sel}'?"):
            self.object_ids = [x for x in self.object_ids if x != sel]
            self.track.pop(sel, None)
            self._refresh_animals_list()
            self._sync_play_animal_dropdown()
            self.log(f"Removed animal: {sel}")
            self._show_frame(self.idx)

    def _refresh_animals_list(self) -> None:
        self.lst_animals.delete(0, "end")
        for a in self.object_ids:
            self.lst_animals.insert("end", a)

    def ui_add_class(self) -> None:
        name = self.ent_class_name.get().strip()
        key = self.ent_class_key.get().strip()
        if not name or not key:
            return
        key = key[0].lower()

        for cn, cd in self.event_classes.items():
            if cd.hotkey.lower() == key:
                messagebox.showerror("Error", f"Class hotkey '{key}' already used by class '{cn}'.")
                return
        if name in self.event_classes:
            messagebox.showerror("Error", "Class name already exists.")
            return

        self.event_classes[name] = EventClass(name=name, hotkey=key)
        self.class_events[name] = {}
        self._refresh_classes_list()
        self.ent_class_name.delete(0, "end")
        self.ent_class_key.delete(0, "end")
        self.log(f"Added class: {name} (hotkey={key})")

    def ui_remove_class(self) -> None:
        sel = self._selected_class_name()
        if not sel:
            return
        if messagebox.askyesno("Remove", f"Remove class '{sel}' and all its events?"):
            self.event_classes.pop(sel, None)
            self.class_events.pop(sel, None)
            self._refresh_classes_list()
            self.lst_events.delete(0, "end")
            self.log(f"Removed class: {sel}")

    def _refresh_classes_list(self) -> None:
        self.lst_classes.delete(0, "end")
        for name, cd in self.event_classes.items():
            self.lst_classes.insert("end", f"{name}  [{cd.hotkey}]")

    def _selected_class_name(self) -> Optional[str]:
        val = self._get_selected_listbox_value(self.lst_classes)
        if not val:
            return None
        return val.split("  [", 1)[0].strip()

    def ui_refresh_events_list(self) -> None:
        cname = self._selected_class_name()
        self.lst_events.delete(0, "end")
        if not cname or cname not in self.class_events:
            return
        for ename, ed in self.class_events[cname].items():
            self.lst_events.insert("end", f"{ename}  [{ed.hotkey}]")

    def ui_add_event(self) -> None:
        self._push_undo("add event")
        cname = self._selected_class_name()
        if not cname:
            messagebox.showerror("Error", "Select a class first.")
            return
        ename = self.ent_event_name.get().strip()
        ekey = self.ent_event_key.get().strip()
        if not ename or not ekey:
            return
        ekey = ekey[0].lower()

        for ed in self.class_events[cname].values():
            if ed.hotkey.lower() == ekey:
                messagebox.showerror("Error", f"Event hotkey '{ekey}' already used in this class.")
                return

        self.class_events[cname][ename] = EventDef(name=ename, hotkey=ekey)
        self.ui_refresh_events_list()
        self.ent_event_name.delete(0, "end")
        self.ent_event_key.delete(0, "end")
        self.log(f"Added event: {cname}/{ename} (hotkey={ekey})")

    def ui_remove_event(self) -> None:
        self._push_undo("remove event")
        cname = self._selected_class_name()
        if not cname:
            return
        val = self._get_selected_listbox_value(self.lst_events)
        if not val:
            return
        ename = val.split("  [", 1)[0].strip()
        if messagebox.askyesno("Remove", f"Remove event '{ename}' from class '{cname}'?"):
            self.class_events[cname].pop(ename, None)
            self.ui_refresh_events_list()
            self.log(f"Removed event: {cname}/{ename}")

    # --------------------------- Presets ---------------------------

    def ui_load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("Text preset", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            animals: List[str] = []
            classes: Dict[str, EventClass] = {}
            events: Dict[str, Dict[str, EventDef]] = {}

            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    up = line.upper()
                    if up.startswith("ANIMAL "):
                        aid = line.split(" ", 1)[1].strip()
                        if aid and aid not in animals:
                            animals.append(aid)
                        continue

                    if up.startswith("CLASS "):
                        rest = line.split(" ", 1)[1].strip()
                        parts = [p.strip() for p in rest.split("|")]
                        if len(parts) != 2:
                            raise ValueError(f"Bad CLASS line: {line}")
                        cname, hotkey = parts[0], parts[1][:1].lower()
                        if not cname or not hotkey:
                            raise ValueError(f"Bad CLASS line: {line}")

                        for oname, oc in classes.items():
                            if oc.hotkey.lower() == hotkey:
                                raise ValueError(
                                    f"Duplicate class hotkey '{hotkey}' for '{cname}' (already used by '{oname}')"
                                )

                        classes[cname] = EventClass(name=cname, hotkey=hotkey)
                        events.setdefault(cname, {})
                        continue

                    if up.startswith("EVENT "):
                        rest = line.split(" ", 1)[1].strip()
                        parts = [p.strip() for p in rest.split("|")]
                        if len(parts) != 3:
                            raise ValueError(f"Bad EVENT line: {line}")
                        cname, ename, hotkey = parts[0], parts[1], parts[2][:1].lower()
                        if not cname or not ename or not hotkey:
                            raise ValueError(f"Bad EVENT line: {line}")

                        events.setdefault(cname, {})
                        for oe in events[cname].values():
                            if oe.hotkey.lower() == hotkey:
                                raise ValueError(f"Duplicate event hotkey '{hotkey}' in class '{cname}'")

                        events[cname][ename] = EventDef(name=ename, hotkey=hotkey)
                        continue

                    raise ValueError(f"Unknown line type: {line}")

            self.object_ids = animals
            self.track = {aid: TrackState() for aid in self.object_ids}

            self.event_classes = classes
            self.class_events = events

            self._refresh_animals_list()
            self._refresh_classes_list()
            self.lst_events.delete(0, "end")
            self.pending_class = None
            self._sync_play_animal_dropdown()

            self.log(
                f"Preset loaded: {os.path.basename(path)} | animals={len(self.object_ids)} classes={len(self.event_classes)}"
            )
            self._show_frame(self.idx)

        except Exception as e:
            messagebox.showerror("Preset error", str(e))

    def _get_selected_listbox_value(self, lb: tk.Listbox) -> Optional[str]:
        sel = lb.curselection()
        if not sel:
            return None
        return lb.get(sel[0])


    # --------------------------- Marker state + logging ---------------------------

    # --------------------------- Undo / Redo ---------------------------

    def _capture_state(self) -> dict:
        """Capture current mutable annotation state."""
        return {
            "idx": int(self.idx),
            "track": copy.deepcopy(self.track),
            "markers_by_frame": copy.deepcopy(self.markers_by_frame),
            "track_log": copy.deepcopy(self.track_log),
            "events_log": copy.deepcopy(self.events_log),
            "pending_class": self.pending_class,
            # "active animal" in this app is driven by the dropdown; there is no
            # persistent attribute called self.active_animal in the release file.
            "active_animal": (self._active_animal() if hasattr(self, "_active_animal") else None),
        }

    def _restore_state(self, state: dict) -> None:
        """Restore previously captured annotation state."""
        self.idx = int(state.get("idx", self.idx))
        self.track = copy.deepcopy(state.get("track", self.track))
        self.markers_by_frame = copy.deepcopy(state.get("markers_by_frame", self.markers_by_frame))
        self.track_log = copy.deepcopy(state.get("track_log", self.track_log))
        self.events_log = copy.deepcopy(state.get("events_log", self.events_log))
        self.pending_class = state.get("pending_class", None)
        # Restore dropdown selection (if present)
        aa = state.get("active_animal", None)
        if aa is not None and hasattr(self, "cmb_active_animal"):
            try:
                self.cmb_active_animal.set(str(aa))
            except Exception:
                pass
        if hasattr(self, "_sync_play_animal_dropdown"):
            self._sync_play_animal_dropdown()
        self._refresh_animals_list()
        self.ui_refresh_events_list()
        self._show_frame(self.idx)
        self._refresh_play_status()

    def _push_undo(self, reason: str) -> None:
        """Push current state to undo stack and clear redo."""
        snap = self._capture_state()
        snap["reason"] = str(reason)
        self._undo_stack.append(snap)
        if len(self._undo_stack) > 200:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def ui_undo(self, _e=None) -> None:
        if not self._undo_stack or self.playing:
            return
        cur = self._capture_state()
        prev = self._undo_stack.pop()
        self._redo_stack.append(cur)
        self._restore_state(prev)
        self.log(f"UNDO: {prev.get('reason','')}")

    def ui_redo(self, _e=None) -> None:
        if not self._redo_stack or self.playing:
            return
        cur = self._capture_state()
        nxt = self._redo_stack.pop()
        self._undo_stack.append(cur)
        self._restore_state(nxt)
        self.log(f"REDO: {nxt.get('reason','')}")


    def _snapshot_frame(self, frame_idx: int) -> None:
        """Store current marker states for the given frame."""
        snap: Dict[str, Tuple[int, int, str, float]] = {}
        for aid, st in self.track.items():
            if st.x is None or st.y is None:
                continue
            snap[aid] = (int(st.x), int(st.y), st.status, float(st.conf))
        self.markers_by_frame[frame_idx] = snap

    def _restore_frame(self, frame_idx: int) -> bool:
        """Restore marker states for the given frame, if present."""
        snap = self.markers_by_frame.get(frame_idx)
        if not snap:
            return False
        for aid, tpl in snap.items():
            x, y, status, conf = tpl
            if aid not in self.track:
                self.track[aid] = TrackState()
            st = self.track[aid]
            st.x, st.y = int(x), int(y)
            st.status = str(status)
            st.conf = float(conf)
        return True

    def _log_track(self, frame_idx: int, aid: str, source: str) -> None:
        """Append a track log row for one animal at one frame."""
        if not self.files or aid not in self.track:
            return
        st = self.track[aid]
        if st.x is None or st.y is None:
            return
        frame_idx = clamp(frame_idx, 0, len(self.files) - 1)
        entry = TrackLogEntry(
            frame_idx=frame_idx,
            t_real=self._t_real_for(frame_idx),
            file_name=self.file_names[frame_idx],
            object_id=aid,
            x=int(st.x),
            y=int(st.y),
            conf=float(st.conf),
            status=str(st.status),
            source=str(source),
        )
        self.track_log.append(entry)

    # --------------------------- Tab 3 (Play) ---------------------------

    def ui_apply_track_params(self) -> None:
        try:
            self.min_conf = float(self.ent_minconf.get())
            self.search_radius = int(float(self.ent_searchr.get()))
        except Exception as e:
            messagebox.showerror("Error", f"Bad tracking params: {e}")
            return
        self.log(f"Tracking params applied: min_conf={self.min_conf:.2f}, search_radius={self.search_radius}px")

    def ui_start_play(self, direction: int) -> None:
        if not self.files:
            return
        self._apply_speed_from_ui()
        self.play_dir = +1 if direction >= 0 else -1
        if not self.playing:
            self.playing = True
            self.log("PLAY " + ("forward" if self.play_dir == 1 else "back"))
            self._refresh_play_status()
            self._play_tick()

    def on_space(self, _e=None) -> None:
        if not self.files:
            return
        if self.playing:
            self.playing = False
            self.after_cancel_safe()
            self.log("PAUSE")
        else:
            self._apply_speed_from_ui()
            self.playing = True
            self.log("RESUME " + ("forward" if self.play_dir == 1 else "back"))
            self._play_tick()
        self._refresh_play_status()

    def _apply_speed_from_ui(self) -> None:
        try:
            self.seconds_per_frame = float(self.ent_speed.get())
            if self.seconds_per_frame <= 0:
                self.seconds_per_frame = 0.05
        except Exception:
            self.seconds_per_frame = 0.20

    def after_cancel_safe(self) -> None:
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _refresh_play_status(self) -> None:
        if not self.files:
            self.lbl_play_status.config(text="(load folder in tab 1)")
            return
        t = self._t_real_for(self.idx)
        fname = self.file_names[self.idx]
        self.lbl_play_status.config(
            text=(
                f"Frame {self.idx+1}/{len(self.files)} | {fname} | t={t} | "
                f"playing={self.playing} | dir={'FWD' if self.play_dir==1 else 'BACK'} | "
                f"pending_class={self.pending_class or '-'}"
            )
        )
        
        
    def _t_real_for(self, idx: int) -> str:

        if self.t_map is not None and 0 <= idx < len(self.t_map):
            return fmt_dt(self.t_map[idx])
        if self.t_start is None:
            return ""
        return fmt_dt(self.t_start + timedelta(minutes=self.step_min * idx))

    def _step_once(self, direction: int) -> None:
        if not self.files or self.playing:
            return
        new_idx = clamp(self.idx + (1 if direction > 0 else -1), 0, len(self.files) - 1)
        self._snapshot_frame(self.idx)
        self.idx = new_idx
        self._restore_frame(self.idx)
        self._show_frame(self.idx)
        self._snapshot_frame(self.idx)
        self._refresh_play_status()

    def _play_tick(self) -> None:
        if not self.playing:
            return

        ok = self._tracking_step(self.play_dir)
        self._show_frame(self.idx)
        self._refresh_play_status()

        if not ok:
            self.playing = False
            self.after_cancel_safe()
            self.log("AUTO-PAUSE (low confidence). Fix marker(s), then Space to resume.")
            self._refresh_play_status()
            return

        nxt = self.idx + self.play_dir
        if nxt < 0 or nxt >= len(self.files):
            self.playing = False
            self.after_cancel_safe()
            self.log("Reached end.")
            self._refresh_play_status()
            return

        self.idx = nxt
        self._show_frame(self.idx)
        self._refresh_play_status()

        delay = int(max(1, self.seconds_per_frame * 1000))
        self.after_id = self.after(delay, self._play_tick)

    # --------------------------- Display ---------------------------

    def _render_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(600, 340, text="Load folder in Tab 1", fill="white", font=("Consolas", 20))

    def _load_bgr(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]
        path = self.files[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        self._cache[idx] = img
        if len(self._cache) > self._cache_max:
            keys = list(self._cache.keys())
            keys.sort(key=lambda k: abs(k - self.idx), reverse=True)
            for k in keys[: len(self._cache) - self._cache_max]:
                self._cache.pop(k, None)
        return img

    def _show_frame(self, idx: int) -> None:
        if not self.files:
            return
        idx = clamp(idx, 0, len(self.files) - 1)

        bgr = self._load_bgr(idx).copy()
        self._current_bgr_cache = bgr

        for aid, st in self.track.items():
            if st.x is None or st.y is None:
                continue
            c = self._color_for_id(aid)
            r = 14 if (self._active_animal() == aid) else 10
            cv2.circle(bgr, (st.x, st.y), r, c, 2, cv2.LINE_AA)
            cv2.putText(bgr, str(aid), (st.x + 10, st.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            if st.status != "tracked":
                cv2.putText(bgr, st.status, (st.x + 10, st.y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

        if (not self.playing) and self.pending_class:
            cv2.putText(
                bgr,
                f"Pending class: {self.pending_class} (press event hotkey)",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2, cv2.LINE_AA
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())

        iw, ih = pil.size
        scale = min(cw / iw, ch / ih) * self.zoom
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self._disp_scale = scale
        offx = (cw - nw) // 2
        offy = (ch - nh) // 2
        if self._zoom_anchor is not None:
            ix, iy, cx, cy = self._zoom_anchor
            offx = int(round(cx - ix * scale))
            offy = int(round(cy - iy * scale))
            self._zoom_anchor = None
            self._view_offx, self._view_offy = offx, offy
        elif (self._view_offx is not None) and (self._view_offy is not None):
            offx, offy = self._view_offx, self._view_offy
        offx = clamp(offx, -nw + 20, cw - 20)
        offy = clamp(offy, -nh + 20, ch - 20)
        self._disp_offx = offx
        self._disp_offy = offy

        pil = pil.resize((nw, nh), resample=pil_bilinear())
        self._tkimg = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(self._disp_offx, self._disp_offy, anchor="nw", image=self._tkimg)

    def _color_for_id(self, aid: str) -> Tuple[int, int, int]:
        h = abs(hash(aid)) % 360
        hsv = np.array([[[h // 2, 220, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # --------------------------- Canvas interaction ---------------------------

    def _canvas_to_img(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        if self._current_bgr_cache is None:
            return None
        ix = int((cx - self._disp_offx) / max(1e-9, self._disp_scale))
        iy = int((cy - self._disp_offy) / max(1e-9, self._disp_scale))
        h, w = self._current_bgr_cache.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def _nearest_marker(self, ix: int, iy: int, max_dist: int = 24) -> Optional[str]:
        best: Optional[str] = None
        bestd = 1e9
        for aid, st in self.track.items():
            if st.x is None or st.y is None:
                continue
            d = (st.x - ix) ** 2 + (st.y - iy) ** 2
            if d < bestd:
                bestd = d
                best = aid
        if best is None:
            return None
        return best if bestd <= max_dist * max_dist else None

    def on_canvas_click(self, e) -> None:
        if self.playing:
            return
        pt = self._canvas_to_img(e.x, e.y)
        if pt is None:
            return
        ix, iy = pt

        near = self._nearest_marker(ix, iy)
        if near:
            self._dragging_id = near
            if not self._drag_undo_pushed:
                self._push_undo(f"move marker {near}")
                self._drag_undo_pushed = True
            self._set_active_animal(near)
            self.log(f"Drag start: {near}")
            return

        if not self.object_ids:
            messagebox.showerror("Error", "Add object IDs in Tab 2 first.")
            return
        self._popup_choose_animal(ix, iy)

    def on_canvas_drag(self, e) -> None:
        if self.playing or not self._dragging_id:
            return
        pt = self._canvas_to_img(e.x, e.y)
        if pt is None:
            return
        ix, iy = pt
        st = self.track[self._dragging_id]
        st.x, st.y = ix, iy
        st.status = "tracked"
        self._update_template_for(self._dragging_id)
        self._show_frame(self.idx)
        self._refresh_play_status()

    def on_canvas_release(self, _e) -> None:
        if self._dragging_id:
            aid = self._dragging_id
            self.log(f"Drag end: {aid}")
            self._log_track(self.idx, aid, source="adjusted")
            self._snapshot_frame(self.idx)
        self._dragging_id = None
        self._drag_undo_pushed = False

    # --------------------------- Zoom ---------------------------

    def set_zoom(self, z: float) -> None:
        z = max(self.zoom_min, min(self.zoom_max, z))
        if abs(z - self.zoom) < 1e-6:
            return
        self.zoom = z
        self.log(f"Zoom: {self.zoom:.2f}x")
        self._show_frame(self.idx)

    def on_mousewheel(self, e) -> None:
        cx, cy = int(getattr(e, "x", 0)), int(getattr(e, "y", 0))
        pt = self._canvas_to_img(cx, cy)
        if e.delta > 0:
            newz = self.zoom * 1.10
        elif e.delta < 0:
            newz = self.zoom / 1.10
        else:
            return
        if pt is not None:
            ix, iy = pt
            self._zoom_anchor = (ix, iy, cx, cy)
        else:
            self._zoom_anchor = None
            self._view_offx, self._view_offy = None, None
        self.set_zoom(newz)

    def on_mousewheel_linux(self, e) -> None:
        cx, cy = int(getattr(e, "x", 0)), int(getattr(e, "y", 0))
        pt = self._canvas_to_img(cx, cy)
        if getattr(e, "num", None) == 4:
            newz = self.zoom * 1.10
        elif getattr(e, "num", None) == 5:
            newz = self.zoom / 1.10
        else:
            return
        if pt is not None:
            ix, iy = pt
            self._zoom_anchor = (ix, iy, cx, cy)
        else:
            self._zoom_anchor = None
            self._view_offx, self._view_offy = None, None
        self.set_zoom(newz)

    def _reset_view_fit(self) -> None:
        self.zoom = 1.0
        self._zoom_anchor = None
        self._view_offx, self._view_offy = None, None
        self._show_frame(self.idx)
        self._refresh_play_status()

    def _popup_choose_animal(self, ix: int, iy: int) -> None:
        top = tk.Toplevel(self)
        top.title("Choose animal ID")
        top.transient(self)
        top.grab_set()

        ttk.Label(top, text="Animal ID:").pack(padx=10, pady=(10, 4))
        cmb = ttk.Combobox(top, values=self.object_ids, state="readonly", width=18)
        cmb.pack(padx=10, pady=(0, 10))
        if self.object_ids:
            cmb.current(0)

        def ok() -> None:
            aid = cmb.get().strip()
            if not aid:
                top.destroy()
                return
            self._push_undo(f"set marker {aid}")
            self._set_marker(aid, ix, iy)
            self._set_active_animal(aid)
            self._log_track(self.idx, aid, source="adjusted")
            self._snapshot_frame(self.idx)
            self.log(f"Marker set: {aid} @ ({ix},{iy}) frame={self.idx} t={self._t_real_for(self.idx)}")
            top.destroy()

        ttk.Button(top, text="OK", command=ok).pack(padx=10, pady=(0, 10))

    def _set_marker(self, aid: str, ix: int, iy: int) -> None:
        if aid not in self.track:
            self.track[aid] = TrackState()
        st = self.track[aid]
        st.x, st.y = ix, iy
        st.status = "tracked"
        self._update_template_for(aid)
        self._show_frame(self.idx)
        self._snapshot_frame(self.idx)
        self._refresh_play_status()

    # --------------------------- Marker active/inactive ---------------------------

    def toggle_active_current_animal(self) -> None:
        self._push_undo("toggle active")
        if self.playing:
            return
        aid = self._active_animal()
        if not aid or aid not in self.track:
            return
        st = self.track[aid]
        if st.status == "inactive":
            st.status = "tracked"
            self._update_template_for(aid)
            self.log(f"Marker ACTIVE: {aid}")
        else:
            st.status = "inactive"
            st.template = None
            st.conf = 0.0
            self.log(f"Marker INACTIVE: {aid}")
        self._show_frame(self.idx)

    # --------------------------- Comments ---------------------------

    def ui_add_comment(self) -> None:
        if self.playing:
            return
        if self.nb.index(self.nb.select()) != 2:
            return

        top = tk.Toplevel(self)
        top.title("Comment")
        top.transient(self)
        top.grab_set()

        ttk.Label(top, text="Enter comment (saved at current frame/time):").pack(padx=10, pady=(10, 4))
        ent = ttk.Entry(top, width=80)
        ent.pack(padx=10, pady=(0, 10))
        ent.focus_set()

        def commit(_e=None) -> None:
            txt = ent.get().strip()
            if txt:
                entry = EventEntry(
                    frame_idx=self.idx,
                    t_real=self._t_real_for(self.idx),
                    file_name=self.file_names[self.idx],
                    object_id=self._active_animal() or "",
                    class_name="COMMENT",
                    event_name="",
                    note=txt,
                )
                self.events_log.append(entry)
                self.log(f"COMMENT: t={entry.t_real} frame={entry.frame_idx} {txt}")
            top.destroy()

        ent.bind("<Return>", commit)
        ttk.Button(top, text="Save", command=commit).pack(padx=10, pady=(0, 10))

    # --------------------------- Templates / active object ---------------------------

    def _update_template_for(self, aid: str) -> None:
        if self._current_bgr_cache is None:
            return
        st = self.track[aid]
        if st.x is None or st.y is None:
            return
        gray = cv2.cvtColor(self._current_bgr_cache, cv2.COLOR_BGR2GRAY)
        h = self.tpl_half
        x0 = clamp(st.x - h, 0, gray.shape[1] - 1)
        y0 = clamp(st.y - h, 0, gray.shape[0] - 1)
        x1 = clamp(st.x + h, 0, gray.shape[1])
        y1 = clamp(st.y + h, 0, gray.shape[0])
        patch = gray[y0:y1, x0:x1]
        if patch.size < 25:
            return
        st.template = patch.copy()

    def _active_animal(self) -> Optional[str]:
        v = self.cmb_active_animal.get().strip()
        return v if v else None

    def _set_active_animal(self, aid: str) -> None:
        if aid in self.object_ids:
            self.cmb_active_animal.set(aid)

    def _sync_play_animal_dropdown(self) -> None:
        self.cmb_active_animal.configure(values=self.object_ids)
        if self.object_ids and not self.cmb_active_animal.get().strip():
            self.cmb_active_animal.set(self.object_ids[0])

    # --------------------------- Hotkeys / Event logging ---------------------------

    def _clear_pending_class(self) -> None:
        if self.pending_class:
            self.log("Pending class cleared")
        self.pending_class = None
        self._show_frame(self.idx)
        self._refresh_play_status()

    def on_key_any(self, e) -> None:
        ks = getattr(e, "keysym", "")
        ctrl = bool(e.state & 0x0004)
        shift = bool(e.state & 0x0001)
        alt = bool((e.state & 0x0008) or (e.state & 0x20000))

        if ks in ("r", "R"):
            if self.nb.index(self.nb.select()) == 2:
                self._reset_view_fit()
                return

        if ks == "grave":
            if ctrl:
                self._grave_down = True
            return

        key = (e.char or "").strip()
        if not key:
            return

        if self.playing:
            return
        if self.nb.index(self.nb.select()) != 2:
            return


        key = key[0].lower()

        if not self.pending_class:
            for cname, cd in self.event_classes.items():
                if cd.hotkey.lower() == key:
                    self.pending_class = cname
                    self.log(f"Class selected: {cname} (press event hotkey)")
                    self._show_frame(self.idx)
                    self._refresh_play_status()
                    return
            return

        cname = self.pending_class
        evs = self.class_events.get(cname, {})
        for ename, ed in evs.items():
            if ed.hotkey.lower() == key:
                aid = self._active_animal()
                if not aid:
                    messagebox.showerror("Error", "Select an active object (dropdown) to attach event.")
                    return
                entry = EventEntry(
                    frame_idx=self.idx,
                    t_real=self._t_real_for(self.idx),
                    file_name=self.file_names[self.idx],
                    object_id=aid,
                    class_name=cname,
                    event_name=ename,
                )
                self.events_log.append(entry)
                self.log(
                    f"EVENT: t={entry.t_real} frame={entry.frame_idx} "
                    f"file={entry.file_name} animal={aid} {cname}/{ename}"
                )
                self.pending_class = None
                self._show_frame(self.idx)
                self._refresh_play_status()
                return


    def on_key_release_any(self, e) -> None:
        ks = getattr(e, "keysym", "")
        if ks == "grave":
            self._grave_down = False


    def _select_animal_slot(self, slot: int) -> None:
        if not self.object_ids:
            return
        if slot < 1 or slot > len(self.object_ids):
            self.log(f"Animal slot {slot} out of range (1..{len(self.object_ids)})")
            return
        aid = self.object_ids[slot - 1]
        self._set_active_animal(aid)
        self.log(f"Active object: {slot} ({aid})")
        self._show_frame(self.idx)
        self._refresh_play_status()
    # --------------------------- Tracking ---------------------------

    def _tracking_step(self, direction: int) -> bool:

        if not self.files:
            return True

        nxt = self.idx + direction
        if nxt < 0 or nxt >= len(self.files):
            return True

        nxtg = self._load_gray(nxt)
        any_low = False

        for aid, st in self.track.items():
            if st.status == "inactive":
                continue
            if st.template is None or st.x is None or st.y is None:
                continue

            r = self.search_radius
            x0 = clamp(st.x - r, 0, nxtg.shape[1] - 1)
            y0 = clamp(st.y - r, 0, nxtg.shape[0] - 1)
            x1 = clamp(st.x + r, 0, nxtg.shape[1])
            y1 = clamp(st.y + r, 0, nxtg.shape[0])

            roi = nxtg[y0:y1, x0:x1]
            if roi.size < 10:
                st.status = "lost"
                st.conf = 0.0
                any_low = True
                continue

            tpl = st.template
            if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
                st.status = "lost"
                st.conf = 0.0
                any_low = True
                continue

            res = cv2.matchTemplate(roi, tpl, self.match_method)
            _, conf, _, loc = cv2.minMaxLoc(res)
            st.conf = float(conf)

            if conf < self.min_conf:
                st.status = "lost"
                any_low = True
                continue

            nx = x0 + loc[0] + tpl.shape[1] // 2
            ny = y0 + loc[1] + tpl.shape[0] // 2
            st.x, st.y = int(nx), int(ny)
            st.status = "tracked"
            self._log_track(nxt, aid, source="detected")

            self._current_bgr_cache = self._load_bgr(nxt).copy()
            self._update_template_for(aid)

        return not any_low

    def _load_gray(self, idx: int) -> np.ndarray:
        bgr = self._load_bgr(idx)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # --------------------------- Save CSV ---------------------------

    def ui_save_csv(self) -> None:
        if not self.folder or not self.files:
            return

        out_tracks = os.path.join(self.folder, "tracks.csv")
        out_events = os.path.join(self.folder, "events.csv")

        with open(out_tracks, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "action_n",
                "frame_idx", "t_real", "file_name",
                "object_id", "x", "y", "conf", "status", "source",
                "ts_sys",
                "camera",
            ])

            action_n = 0

            if self.track_log:
                for tr in self.track_log:
                    action_n += 1
                    w.writerow([
                        action_n,
                        tr.frame_idx, tr.t_real, tr.file_name,
                        tr.object_id, tr.x, tr.y, f"{tr.conf:.3f}", tr.status, tr.source,
                        datetime.now().isoformat(timespec="milliseconds"),
                        self.camera_name,
                    ])
            else:
                for aid, st in self.track.items():
                    if st.x is None or st.y is None:
                        continue
                    action_n += 1
                    w.writerow([
                        action_n,
                        self.idx, self._t_real_for(self.idx), self.file_names[self.idx],
                        aid, st.x, st.y, f"{st.conf:.3f}", st.status, "",
                        datetime.now().isoformat(timespec="milliseconds"),
                        self.camera_name,
                    ])

        with open(out_events, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "frame_idx", "t_real", "file_name",
                "object_id", "class_name", "event_name", "note", "camera",
            ])
            for e in self.events_log:
                w.writerow([
                    e.frame_idx, e.t_real, e.file_name,
                    e.object_id, e.class_name, e.event_name, e.note, self.camera_name,
                ])

        self.log(f"Saved: {out_tracks}")
        self.log(f"Saved: {out_events}")
        messagebox.showinfo("Saved", "tracks.csv and events.csv saved to folder.")


if __name__ == "__main__":
    app = SealGUI()
    app.mainloop()