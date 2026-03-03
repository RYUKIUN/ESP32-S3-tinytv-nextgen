import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
import json
import serial
from queue import Queue

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240          # ILI9341 landscape

# Tile constants — MUST match main.cpp
CHUNK_DATA_SIZE  = 1400          # bytes of JPEG payload per UDP packet
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120

# Tile layout (x, y offsets in the 320x240 frame):
#   [0: TL]  [1: TR]
#   [2: BL]  [3: BR]
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]

# Per-tile JPEG size cap — matches MAX_TILE_JPEG in main.cpp
MAX_TILE_JPEG = 33600            # 24 × 1400

DEFAULT_FPS   = 30
WINDOW_NAME   = "Stream Control"
PROFILE_FILE  = "stream_profiles.json"
SHARP_BIAS    = -0.04

# Transport selection
USE_USB       = True             # True = USB CDC bulk, False = UDP WiFi
USB_PORT      = "COM6"           # Set to your ESP32-S3 COM port on Windows
USB_BAUD      = 2000000          # Effective rate is USB FS; baud is advisory

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
frame_queue     = Queue(maxsize=1)
settings_lock   = threading.Lock()
stream_profiles = {}
current_mode    = 0
stop_event      = threading.Event()
_frame_id       = 0              # rolling 0-255 frame counter

STATIC_NOISE = np.zeros((ESP_H, ESP_W, 3), dtype=np.int8)
cv2.randn(STATIC_NOISE, 0, 2)

# ─────────────────────────────────────────────
#  SYSTEM HELPERS
# ─────────────────────────────────────────────
def set_high_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeBeginPeriod(1)
        except: pass

def reset_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeEndPeriod(1)
        except: pass

def set_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt': p.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:                p.nice(-10)
    except: pass

def get_mouse_pos():
    if os.name == 'nt':
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    return 0, 0

# ─────────────────────────────────────────────
#  PROFILE PERSISTENCE
# ─────────────────────────────────────────────
def load_profiles():
    global stream_profiles
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, 'r') as f:
                stream_profiles = json.load(f)
            print(f"[*] Loaded {len(stream_profiles)} profiles.")
        except: stream_profiles = {}

def save_profiles():
    with settings_lock:
        try:
            with open(PROFILE_FILE, 'w') as f:
                json.dump(stream_profiles, f, indent=4)
            print(f"\n[!] Profiles saved → {os.path.abspath(PROFILE_FILE)}")
        except Exception as e:
            print(f"Save Error: {e}")

# ─────────────────────────────────────────────
#  CONTENT ANALYSIS  (lightweight — runs every frame)
# ─────────────────────────────────────────────
def get_scene_key(frame_small):
    """Return (key_str, variance, edge_density) for scene-profile lookup.
    Uses std-dev and Laplacian variance — both O(N), no convolution overhead."""
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    var  = cv2.meanStdDev(gray)[1][0][0]
    edge_density = cv2.Laplacian(gray, cv2.CV_64F).var()
    v_cat = int(np.clip(var / 32, 0, 3))
    e_cat = int(np.clip(edge_density / 100, 0, 3))
    return f"v{v_cat}_e{e_cat}", var, edge_density

# ─────────────────────────────────────────────
#  LIGHTWEIGHT QUALITY METRIC  (replaces SSIM)
# ─────────────────────────────────────────────
def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB.
    Pure NumPy: O(N) with no kernel/window overhead.  Typical range 20–50 dB.
    Higher = better fidelity."""
    mse = np.mean((ref.astype(np.float32) - test.astype(np.float32)) ** 2)
    if mse < 1e-8:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))

# ─────────────────────────────────────────────
#  TRANSMIT — chunked UDP / USB
# ─────────────────────────────────────────────
# Pre-allocated send buffer — avoids a `bytes` heap allocation per packet.
_send_buf  = bytearray(8 + CHUNK_DATA_SIZE)
_send_view = memoryview(_send_buf)

def send_tiles(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, sub_flag: int, pacing_s: float = 0.0) -> int:
    """
    Split 320×240 frame into 4 independent 160×120 tiles, JPEG-encode each,
    send each tile's chunks over UDP.

    Returns total bytes sent across all tiles (for display stats — free).

    Packet header (8 bytes):
      [0xAA, 0xBB, frame_id, tile_id, chunk_id, total_chunks, size_hi, size_lo]

    Tile independence means a single lost packet degrades one tile for one frame
    (20% of screen) rather than corrupting the whole frame.
    """
    global _frame_id

    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    dest      = (target_ip, PORT)
    total_bytes = 0

    for tId in range(NUM_TILES):
        x, y  = TILE_X[tId], TILE_Y[tId]
        tile  = frame_bgr[y:y+TILE_H, x:x+TILE_W]

        _, enc = cv2.imencode('.jpg', tile,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                               int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        jpeg_bytes = bytes(enc)          # one copy — reused for all chunks below
        total_len  = len(jpeg_bytes)

        if total_len > MAX_TILE_JPEG:
            print(f"[WARN] Tile {tId} JPEG {total_len}B > MAX_TILE_JPEG — lower quality")
            continue

        total_bytes += total_len
        num_chunks   = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
        size_hi      = (total_len >> 8) & 0xFF
        size_lo      = total_len & 0xFF

        for cId in range(num_chunks):
            offset = cId * CHUNK_DATA_SIZE
            clen   = min(CHUNK_DATA_SIZE, total_len - offset)

            # Fill pre-allocated buffer in-place — no per-packet heap allocation
            _send_buf[0] = 0xAA
            _send_buf[1] = 0xBB
            _send_buf[2] = frame_id
            _send_buf[3] = tId
            _send_buf[4] = cId
            _send_buf[5] = num_chunks
            _send_buf[6] = size_hi
            _send_buf[7] = size_lo
            _send_buf[8:8+clen] = jpeg_bytes[offset:offset+clen]

            # Retry on WSAEWOULDBLOCK / EAGAIN — send buffer temporarily full
            while True:
                try:
                    sock.sendto(_send_view[:8+clen], dest)
                    break
                except BlockingIOError:
                    time.sleep(0.0005)

            if pacing_s > 0:
                time.sleep(pacing_s)

    return total_bytes


def send_tiles_usb(ser: serial.Serial, frame_bgr: np.ndarray,
                   quality: int, sub_flag: int) -> int:
    """
    Same packet format as UDP, but written to a USB CDC serial stream.
    Data header (8 bytes) + payload bytes are concatenated back-to-back.
    """
    global _frame_id

    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    total_bytes = 0

    for tId in range(NUM_TILES):
        x, y  = TILE_X[tId], TILE_Y[tId]
        tile  = frame_bgr[y:y+TILE_H, x:x+TILE_W]

        _, enc = cv2.imencode('.jpg', tile,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                               int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        jpeg_bytes = bytes(enc)
        total_len  = len(jpeg_bytes)

        if total_len > MAX_TILE_JPEG:
            print(f"[WARN] Tile {tId} JPEG {total_len}B > MAX_TILE_JPEG — lower quality")
            continue

        total_bytes += total_len
        num_chunks   = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
        size_hi      = (total_len >> 8) & 0xFF
        size_lo      = total_len & 0xFF

        for cId in range(num_chunks):
            offset = cId * CHUNK_DATA_SIZE
            clen   = min(CHUNK_DATA_SIZE, total_len - offset)

            _send_buf[0] = 0xAA
            _send_buf[1] = 0xBB
            _send_buf[2] = frame_id
            _send_buf[3] = tId
            _send_buf[4] = cId
            _send_buf[5] = num_chunks
            _send_buf[6] = size_hi
            _send_buf[7] = size_lo
            _send_buf[8:8+clen] = jpeg_bytes[offset:offset+clen]

            ser.write(_send_view[:8+clen])

    return total_bytes

# ─────────────────────────────────────────────
#  BACKGROUND EXHAUSTIVE PROFILER
# ─────────────────────────────────────────────
# Runs every ~10 s in a daemon thread.  Finds the best dither/sharp/sub/quality
# combination for the current scene type and saves it as a profile.
#
# Quality metric: PSNR (replaces SSIM).
#   PSNR is O(N) — just MSE + log.  No sliding-window convolution.
#   Normalised to ~0-1 range (÷50) so the scoring arithmetic is identical
#   to the old SSIM path; sharp_bias / subsampling penalties apply unchanged.
#
# Works on a single representative tile (TL) rather than the full frame:
#   • Smaller array → faster encode/decode per iteration
#   • MAX_TILE_JPEG is the real constraint the ESP cares about
#   • All tiles share the same encoding parameters anyway
def background_profiler():
    global current_mode
    while not stop_event.is_set():
        # Idle for 10 s between profiling runs
        for _ in range(100):
            if stop_event.is_set(): return
            time.sleep(0.1)

        if current_mode == 0 or frame_queue.empty():
            continue

        raw_frame = frame_queue.get()
        full      = cv2.resize(raw_frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
        key, var, edges = get_scene_key(full)

        # Reference: top-left tile only — representative and fast
        ref_tile = full[TILE_Y[0]:TILE_Y[0]+TILE_H, TILE_X[0]:TILE_X[0]+TILE_W].copy()

        best_score = -999.0
        best_cfg   = {"dither": 0, "sharp": 0.0, "sub": 444, "q": 70}

        for d in [0, 2, 4]:
            for s in [0.0, 0.1, 0.2, 0.3]:
                for subsampling in [444, 420]:
                    test_tile = ref_tile.copy()

                    if d > 0:
                        # Crop noise to tile size for the profiler pass
                        noise_tile = (STATIC_NOISE[TILE_Y[0]:TILE_Y[0]+TILE_H,
                                                   TILE_X[0]:TILE_X[0]+TILE_W]
                                      .astype(np.float32) * (d / 2.0)).astype(np.int8)
                        test_tile = cv2.add(test_tile, noise_tile, dtype=cv2.CV_8U)

                    if s > 0:
                        k = np.array([[0, -s, 0],
                                      [-s, 1 + 4*s, -s],
                                      [0, -s, 0]])
                        test_tile = cv2.filter2D(test_tile, -1, k)

                    sub_flag = (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
                                if subsampling == 444
                                else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)

                    # Binary search: highest quality whose JPEG fits MAX_TILE_JPEG
                    low_q, high_q, found_q = 10, 95, 10
                    while low_q <= high_q:
                        mid_q = (low_q + high_q) // 2
                        _, enc = cv2.imencode('.jpg', test_tile,
                                             [int(cv2.IMWRITE_JPEG_QUALITY), mid_q,
                                              int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
                        if len(enc) <= MAX_TILE_JPEG:
                            found_q = mid_q; low_q = mid_q + 1
                        else:
                            high_q  = mid_q - 1

                    # Decode at found quality and score with PSNR
                    _, final_enc = cv2.imencode('.jpg', test_tile,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), found_q,
                                                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
                    decoded = cv2.imdecode(final_enc, cv2.IMREAD_COLOR)

                    # PSNR normalised to ~0-1 (50 dB ≈ excellent for this display)
                    visual_score = psnr(ref_tile, decoded) / 50.0

                    # Apply the same biases as before
                    adjusted = visual_score - (s * SHARP_BIAS)
                    if subsampling == 420:
                        adjusted -= 0.02

                    if adjusted > best_score:
                        best_score = adjusted
                        best_cfg   = {"dither": d, "sharp": s,
                                      "sub": subsampling, "q": found_q}

        with settings_lock:
            stream_profiles[key] = best_cfg
            print(f"[PROFILE] {key} → D:{best_cfg['dither']} "
                  f"S:{best_cfg['sharp']} Sub:{best_cfg['sub']} Q:{best_cfg['q']} "
                  f"(score={best_score:.4f})")

# ─────────────────────────────────────────────
#  CAPTURE WORKER
# ─────────────────────────────────────────────
def select_display_mss():
    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors):
            if i == 0: continue
            if mon["width"] < 1920: return i
        return 1

def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            sct_img = sct.grab(monitor)
            frame   = (np.frombuffer(sct_img.raw, dtype=np.uint8)
                       .reshape((monitor["height"], monitor["width"], 4))[:, :, :3]
                       .copy())
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame)

# ─────────────────────────────────────────────
#  STAT PACKET PARSER
# ─────────────────────────────────────────────
def parse_esp_stats(raw: str) -> dict:
    """Parse the compact stat string from ESP32:
       FPS:X|TEMP:X|JIT:X|DEC:X|DROP:X|SRAM:X/X|PSRAM:X/X
    Returns a dict with string values, or '?' for missing fields."""
    fields = {}
    for token in raw.split('|'):
        if ':' in token:
            k, _, v = token.partition(':')
            fields[k.strip()] = v.strip()
    return fields

def _diag_color(val_str: str, warn: float, err: float):
    """Return BGR colour tuple based on numeric thresholds."""
    try:
        # Strip any unit suffix (ms, KB, °C, %)
        v = float(''.join(c for c in val_str if c in '0123456789.-'))
        if v >= err:  return (0,   0, 255)   # red
        if v >= warn: return (0, 165, 255)   # orange
    except: pass
    return (0, 255, 0)                        # green

def _sram_color(free_kb_str: str):
    """Colour SRAM/PSRAM free field — warns when headroom is low."""
    try:
        free = float(free_kb_str.split('/')[0])
        if free < 20:  return (0,   0, 255)
        if free < 50:  return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  STREAM + UI (UDP)
# ─────────────────────────────────────────────
def stream_mss_udp(target_ip: str, monitor_idx: int):
    global current_mode

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 480, 560)
    cv2.createTrackbar("Max FPS",        WINDOW_NAME, DEFAULT_FPS, 60, lambda x: None)
    cv2.createTrackbar("Base Qual",      WINDOW_NAME, 70, 95,       lambda x: None)
    cv2.createTrackbar("Mode FAST/TUNE", WINDOW_NAME, 0,  1,        lambda x: None)
    cv2.createTrackbar("Debug Info",     WINDOW_NAME, 1,  1,        lambda x: None)
    # Pacing: 0-10 = 0-1000 µs between packets within a tile.
    # Start at 0, increase if tiles show corrupt-marker errors.
    cv2.createTrackbar("Pacing x100us", WINDOW_NAME, 0, 10, lambda x: None)

    threading.Thread(target=capture_worker,    args=(monitor_idx,), daemon=True).start()
    threading.Thread(target=background_profiler,                     daemon=True).start()

    with mss.mss() as sct:
        m = sct.monitors[monitor_idx]
        m_left, m_top, m_w, m_h = m["left"], m["top"], m["width"], m["height"]

    latest_esp_stats  = {}            # parsed dict from last ESP stat packet
    latest_esp_raw    = "Waiting for ESP32-S3..."
    last_debug_send   = 0
    last_frame_bytes  = 0

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()

            fps          = cv2.getTrackbarPos("Max FPS",        WINDOW_NAME)
            user_qual    = cv2.getTrackbarPos("Base Qual",      WINDOW_NAME)
            current_mode = cv2.getTrackbarPos("Mode FAST/TUNE", WINDOW_NAME)
            debug_state  = cv2.getTrackbarPos("Debug Info",     WINDOW_NAME)
            pacing_us    = cv2.getTrackbarPos("Pacing x100us",  WINDOW_NAME)
            pacing_s     = pacing_us * 0.0001

            # ── Receive stat packets from ESP ───────────────────────────────
            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        raw = data[2:].decode('utf-8', errors='ignore')
                        latest_esp_raw   = raw
                        latest_esp_stats = parse_esp_stats(raw)
            except: pass

            # ── Send debug toggle (every 500 ms) ───────────────────────────
            if time.time() - last_debug_send > 0.5:
                sock.sendto(bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue

            frame = frame_queue.get()

            # ── Draw cursor ─────────────────────────────────────────────────
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), 8, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), 5, (0,   0, 255),  -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            key, var, edges = get_scene_key(resized)

            # ── Encoding parameters ────────────────────────────────────────
            d_amt, s_amt, sub, q_final = 0, 0.0, 444, user_qual

            with settings_lock:
                if key in stream_profiles:
                    d_amt   = stream_profiles[key]["dither"]
                    s_amt   = stream_profiles[key]["sharp"]
                    sub     = stream_profiles[key]["sub"]
                    q_final = min(user_qual, stream_profiles[key]["q"])

            if current_mode == 0 or key not in stream_profiles:
                d_amt = 2    if var   < 15  else 0
                s_amt = 0.15 if edges > 300 else 0.0
                sub   = 420

            if d_amt > 0:
                n       = (STATIC_NOISE.astype(np.float32) * (d_amt / 2.0)).astype(np.int8)
                resized = cv2.add(resized, n, dtype=cv2.CV_8U)
            if s_amt > 0:
                k       = np.array([[0, -s_amt, 0],
                                    [-s_amt, 1 + 4*s_amt, -s_amt],
                                    [0, -s_amt, 0]])
                resized = cv2.filter2D(resized, -1, k)

            sub_flag = (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
                        if sub == 420
                        else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)

            # ── Transmit — send_tiles returns total bytes; no re-encoding ──
            last_frame_bytes = send_tiles(
                sock, target_ip, resized, q_final, sub_flag, pacing_s=pacing_s)

            # ── Preview window ──────────────────────────────────────────────
            preview = cv2.resize(resized, (480, 360), interpolation=cv2.INTER_NEAREST)
            f = latest_esp_stats   # shorthand

            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (480, 360), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, 0.85, preview, 0.15, 0)

                # ── Compact diagnostic dashboard ────────────────────────────
                # Fields: FPS, TEMP, JIT, DEC, DROP, SRAM (free/total), PSRAM (free/total)
                dashboard = [
                    (f"FPS  : {f.get('FPS',  '?'):>8}",
                     _diag_color(f.get('FPS', '0'),  15, 5)),   # warn <15, err <5

                    (f"TEMP : {f.get('TEMP', '?'):>7} C",
                     _diag_color(f.get('TEMP','0'),  70, 85)),  # warn >70, err >85

                    (f"JIT  : {f.get('JIT',  '?'):>7} ms",
                     _diag_color(f.get('JIT', '0'),  10, 30)),  # warn >10, err >30

                    (f"DEC  : {f.get('DEC',  '?'):>7} us",
                     _diag_color(f.get('DEC', '0'),  8000, 15000)),

                    (f"DROP : {f.get('DROP', '?'):>8}",
                     _diag_color(f.get('DROP','0'),  1, 5)),    # warn >1/s, err >5/s

                    (f"SRAM : {f.get('SRAM',  '?/?' ):>11} KB",
                     _sram_color(f.get('SRAM', '999/1'))),

                    (f"PSRAM: {f.get('PSRAM', '?/?' ):>11} KB",
                     _sram_color(f.get('PSRAM','999/1'))),
                ]

                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
                    y += 22
            else:
                per_tile  = last_frame_bytes // NUM_TILES if last_frame_bytes else 0
                pkts_per  = (per_tile + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE if per_tile else 0
                info = (f"Total:{last_frame_bytes}B  PerTile:~{per_tile}B/{pkts_per}pkts "
                        f"Q:{q_final} Sub:{sub} Pace:{pacing_us*100}us "
                        f"{'[Tune]' if current_mode else '[Fast]'}")
                cv2.putText(preview, info, (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed  = time.perf_counter() - t_start
            wait_ms  = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
        save_profiles()
        cv2.destroyAllWindows()
        sock.close()

# ─────────────────────────────────────────────
#  STREAM + UI (USB CDC)
# ─────────────────────────────────────────────
def stream_mss_usb(port: str, monitor_idx: int):
    global current_mode

    ser = serial.Serial(port, USB_BAUD, timeout=0)
    print(f"[*] USB stream on {port} @ {USB_BAUD}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 480, 560)
    cv2.createTrackbar("Max FPS",        WINDOW_NAME, DEFAULT_FPS, 60, lambda x: None)
    cv2.createTrackbar("Base Qual",      WINDOW_NAME, 70, 95,       lambda x: None)
    cv2.createTrackbar("Mode FAST/TUNE", WINDOW_NAME, 0,  1,        lambda x: None)
    cv2.createTrackbar("Debug Info",     WINDOW_NAME, 0,  1,        lambda x: None)

    threading.Thread(target=capture_worker,    args=(monitor_idx,), daemon=True).start()
    threading.Thread(target=background_profiler,                     daemon=True).start()

    with mss.mss() as sct:
        m = sct.monitors[monitor_idx]
        m_left, m_top, m_w, m_h = m["left"], m["top"], m["width"], m["height"]

    latest_esp_stats  = {}
    latest_esp_raw    = "USB mode"
    last_frame_bytes  = 0

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()

            fps          = cv2.getTrackbarPos("Max FPS",        WINDOW_NAME)
            user_qual    = cv2.getTrackbarPos("Base Qual",      WINDOW_NAME)
            current_mode = cv2.getTrackbarPos("Mode FAST/TUNE", WINDOW_NAME)
            debug_state  = cv2.getTrackbarPos("Debug Info",     WINDOW_NAME)

            # Read any pending stats line from ESP (optional)
            if debug_state == 1:
                try:
                    raw = ser.read(256)
                    if raw.startswith(b"\xAB\xCD"):
                        txt = raw[2:].decode("utf-8", errors="ignore")
                        latest_esp_raw   = txt
                        latest_esp_stats = parse_esp_stats(txt)
                except:  # noqa: E722
                    pass

            if frame_queue.empty(): continue

            frame = frame_queue.get()

            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), 8, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), 5, (0,   0, 255),  -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            key, var, edges = get_scene_key(resized)

            d_amt, s_amt, sub, q_final = 0, 0.0, 444, user_qual

            with settings_lock:
                if key in stream_profiles:
                    d_amt   = stream_profiles[key]["dither"]
                    s_amt   = stream_profiles[key]["sharp"]
                    sub     = stream_profiles[key]["sub"]
                    q_final = min(user_qual, stream_profiles[key]["q"])

            if current_mode == 0 or key not in stream_profiles:
                d_amt = 2    if var   < 15  else 0
                s_amt = 0.15 if edges > 300 else 0.0
                sub   = 420

            if d_amt > 0:
                n       = (STATIC_NOISE.astype(np.float32) * (d_amt / 2.0)).astype(np.int8)
                resized = cv2.add(resized, n, dtype=cv2.CV_8U)
            if s_amt > 0:
                k       = np.array([[0, -s_amt, 0],
                                    [-s_amt, 1 + 4*s_amt, -s_amt],
                                    [0, -s_amt, 0]])
                resized = cv2.filter2D(resized, -1, k)

            sub_flag = (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
                        if sub == 420
                        else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)

            last_frame_bytes = send_tiles_usb(ser, resized, q_final, sub_flag)

            preview = cv2.resize(resized, (480, 360), interpolation=cv2.INTER_NEAREST)
            f = latest_esp_stats

            if debug_state == 1 and f:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (480, 360), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, 0.85, preview, 0.15, 0)

                dashboard = [
                    (f"FPS  : {f.get('FPS',  '?'):>8}",
                     _diag_color(f.get('FPS', '0'),  15, 5)),
                    (f"TEMP : {f.get('TEMP', '?'):>7} C",
                     _diag_color(f.get('TEMP','0'),  70, 85)),
                    (f"JIT  : {f.get('JIT',  '?'):>7} ms",
                     _diag_color(f.get('JIT', '0'),  10, 30)),
                    (f"DEC  : {f.get('DEC',  '?'):>7} us",
                     _diag_color(f.get('DEC', '0'),  8000, 15000)),
                    (f"DROP : {f.get('DROP', '?'):>8}",
                     _diag_color(f.get('DROP','0'),  1, 5)),
                    (f"SRAM : {f.get('SRAM',  '?/?' ):>11} KB",
                     _sram_color(f.get('SRAM', '999/1'))),
                    (f"PSRAM: {f.get('PSRAM', '?/?' ):>11} KB",
                     _sram_color(f.get('PSRAM','999/1'))),
                ]

                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
                    y += 22
            else:
                per_tile  = last_frame_bytes // NUM_TILES if last_frame_bytes else 0
                info = (f"Total:{last_frame_bytes}B  PerTile:~{per_tile}B  "
                        f"Q:{q_final} Sub:{sub} [USB]")
                cv2.putText(preview, info, (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed  = time.perf_counter() - t_start
            wait_ms  = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
        save_profiles()
        cv2.destroyAllWindows()
        ser.close()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def quick_find_esp(timeout=5.0) -> str | None:
    """Listen for the ESP32's UDP beacon and return its IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', PORT))
    s.settimeout(timeout)
    print(f"[*] Waiting {timeout}s for ESP32 beacon on port {PORT}…")
    try:
        data, addr = s.recvfrom(128)
        if b"ESP32_READY" in data:
            print(f"[*] Found ESP32 at {addr[0]}")
            return addr[0]
    except socket.timeout:
        print("[!] No beacon received. Hardcode the IP below if needed.")
    finally:
        s.close()
    return None


if __name__ == "__main__":
    set_high_priority()
    set_high_resolution_timer()
    load_profiles()

    if USE_USB:
        stream_mss_usb(USB_PORT, select_display_mss())
    else:
        # ── Auto-discover or hardcode ──────────────────────
        ip = quick_find_esp(timeout=5.0)
        # ip = "192.168.x.x"   # ← uncomment and set if auto-discovery fails

        if ip:
            stream_mss_udp(ip, select_display_mss())
        else:
            print("[!] Could not find ESP32. Stream aborted.")

    reset_resolution_timer()