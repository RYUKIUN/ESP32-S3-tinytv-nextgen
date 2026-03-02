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
from queue import Queue
from skimage.metrics import structural_similarity as ssim

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT        = 12345
ESP_W, ESP_H = 320, 240          # ← New resolution (ILI9341 landscape)

# Tile constants — MUST match main.cpp
CHUNK_DATA_SIZE  = 1400   # bytes of JPEG payload per UDP packet
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120

# Tile layout (x, y offsets in the 320x240 frame):
#   [0: TL]  [1: TR]
#   [2: BL]  [3: BR]
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]

# Per-tile JPEG size cap. 160x120 at quality 95 is ~12 KB worst case.
MAX_TILE_JPEG = 33600   # 24 × 1400 — matches MAX_TILE_CHUNKS in main.cpp

DEFAULT_FPS   = 30
WINDOW_NAME   = "Stream Control"
PROFILE_FILE  = "stream_profiles.json"

# Sharpness bias (see original code comments)
SHARP_BIAS = -0.04

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
frame_queue    = Queue(maxsize=1)
settings_lock  = threading.Lock()
stream_profiles = {}
current_mode   = 0
stop_event     = threading.Event()
_frame_id      = 0               # rolling 0-255 frame counter

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
        else: p.nice(-10)
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
#  CONTENT ANALYSIS
# ─────────────────────────────────────────────
def get_scene_key(frame_small):
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    var  = cv2.meanStdDev(gray)[1][0][0]
    edge_density = cv2.Laplacian(gray, cv2.CV_64F).var()
    v_cat = int(np.clip(var / 32, 0, 3))
    e_cat = int(np.clip(edge_density / 100, 0, 3))
    return f"v{v_cat}_e{e_cat}", var, edge_density

# ─────────────────────────────────────────────
#  FEC CHUNKING
# ─────────────────────────────────────────────

def send_tiles(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, sub_flag: int, pacing_s: float = 0.0):
    """
    Split the 320x240 frame into 4 independent 160x120 tiles, encode each
    as a separate JPEG, and send each tile's chunks independently.

    Packet header (8 bytes):
      [0xAA, 0xBB, frame_id, tile_id, chunk_id, total_chunks, size_hi, size_lo]

    Why tiles beat whole-frame chunking:
    - A 160x120 tile at quality 70 is ~3-6 KB = 3-5 UDP packets.
    - Losing 1 packet = 1 tile skips an update (20% of screen for one frame).
    - Old approach: losing 1 packet = whole frame corrupt (100% screen).
    - Each tile is independently valid JPEG — no cross-tile dependency.
    """
    global _frame_id

    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    dest      = (target_ip, PORT)

    for tId in range(NUM_TILES):
        # Crop tile from full frame
        x, y   = TILE_X[tId], TILE_Y[tId]
        tile   = frame_bgr[y:y+TILE_H, x:x+TILE_W]

        # Encode tile to JPEG
        _, enc = cv2.imencode('.jpg', tile,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                               int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        jpeg_bytes = enc.tobytes()
        total_len  = len(jpeg_bytes)

        if total_len > MAX_TILE_JPEG:
            print(f"[WARN] Tile {tId} JPEG {total_len}B exceeds MAX_TILE_JPEG — lower quality")
            continue

        # Chunk and send
        num_chunks = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
        size_hi    = (total_len >> 8) & 0xFF
        size_lo    = total_len & 0xFF

        for cId in range(num_chunks):
            chunk  = jpeg_bytes[cId * CHUNK_DATA_SIZE : (cId + 1) * CHUNK_DATA_SIZE]
            header = bytes([0xAA, 0xBB, frame_id, tId, cId, num_chunks, size_hi, size_lo])
            # Retry on WSAEWOULDBLOCK (WinError 10035) — send buffer temporarily full
            while True:
                try:
                    sock.sendto(header + chunk, dest)
                    break
                except BlockingIOError:
                    time.sleep(0.0005)  # wait 500µs and retry
            if pacing_s > 0:
                time.sleep(pacing_s)


# ─────────────────────────────────────────────
#  BACKGROUND EXHAUSTIVE PROFILER
# ─────────────────────────────────────────────
def background_profiler():
    global current_mode
    while not stop_event.is_set():
        for _ in range(100):
            if stop_event.is_set(): return
            time.sleep(0.1)

        if current_mode == 0 or frame_queue.empty(): continue

        raw_frame = frame_queue.get()
        target    = cv2.resize(raw_frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
        key, var, edges = get_scene_key(target)

        best_score = -999.0
        best_cfg   = {"dither": 0, "sharp": 0.0, "sub": 444, "q": 70}

        for d in [0, 2, 4]:
            for s in [0.0, 0.1, 0.2, 0.3]:
                for subsampling in [444, 420]:
                    test_img = target.copy()
                    if d > 0:
                        n = (STATIC_NOISE.astype(np.float32) * (d / 2.0)).astype(np.int8)
                        test_img = cv2.add(test_img, n, dtype=cv2.CV_8U)
                    if s > 0:
                        k = np.array([[0, -s, 0], [-s, 1 + 4*s, -s], [0, -s, 0]])
                        test_img = cv2.filter2D(test_img, -1, k)

                    sub_flag = (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
                                if subsampling == 444
                                else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)

                    # Binary search: highest quality that fits MAX_FRAME_SIZE
                    low_q, high_q, found_q = 10, 95, 10
                    while low_q <= high_q:
                        mid_q = (low_q + high_q) // 2
                        _, enc = cv2.imencode('.jpg', test_img,
                                             [int(cv2.IMWRITE_JPEG_QUALITY), mid_q,
                                              int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
                        if len(enc) <= MAX_FRAME_SIZE:
                            found_q = mid_q; low_q = mid_q + 1
                        else:
                            high_q = mid_q - 1

                    _, final_enc = cv2.imencode('.jpg', test_img,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), found_q,
                                                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
                    decoded = cv2.imdecode(final_enc, 1)
                    visual_score = ssim(target, decoded, channel_axis=2)

                    adjusted = visual_score - (s * SHARP_BIAS)
                    if subsampling == 420: adjusted -= 0.02

                    if adjusted > best_score:
                        best_score = adjusted
                        best_cfg   = {"dither": d, "sharp": s,
                                      "sub": subsampling, "q": found_q}

        with settings_lock:
            stream_profiles[key] = best_cfg
            print(f"[EXHAUSTIVE] {key} → D:{best_cfg['dither']} "
                  f"S:{best_cfg['sharp']} Sub:{best_cfg['sub']} Q:{best_cfg['q']}")


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
#  STREAM + UI
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
    # Pacing: 0-10 = 0-1000µs between packets within a tile.
    # Start at 0, increase only if tiles show corrupt markers.
    cv2.createTrackbar("Pacing x100us", WINDOW_NAME, 0, 10, lambda x: None)

    threading.Thread(target=capture_worker,    args=(monitor_idx,), daemon=True).start()
    threading.Thread(target=background_profiler,                     daemon=True).start()

    with mss.mss() as sct:
        m = sct.monitors[monitor_idx]
        m_left, m_top, m_w, m_h = m["left"], m["top"], m["width"], m["height"]

    latest_esp_log    = "Waiting for ESP32-S3..."
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
            pacing_us  = cv2.getTrackbarPos("Pacing x100us", WINDOW_NAME)
            pacing_s   = pacing_us * 0.0001

            # Receive stat packets from ESP
            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        latest_esp_log = data[2:].decode('utf-8', errors='ignore')
            except: pass

            # Send debug toggle
            if time.time() - last_debug_send > 0.5:
                sock.sendto(bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue
            frame = frame_queue.get()

            # Draw cursor
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), 8, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), 5, (0, 0, 255),     -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            key, var, edges = get_scene_key(resized)

            # ── Encoding parameters ────────────────────
            d_amt, s_amt, sub, q_final = 0, 0.0, 444, user_qual

            with settings_lock:
                if key in stream_profiles:
                    d_amt   = stream_profiles[key]["dither"]
                    s_amt   = stream_profiles[key]["sharp"]
                    sub     = stream_profiles[key]["sub"]
                    q_final = min(user_qual, stream_profiles[key]["q"])

            if current_mode == 0 or key not in stream_profiles:
                d_amt   = 2   if var   < 15  else 0
                s_amt   = 0.15 if edges > 300 else 0.0
                sub     = 420

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

            # ── Transmit as 4 independent tiles ────────
            # send_tiles handles per-tile JPEG encoding internally.
            # last_frame_bytes tracks total bytes across all tiles for the UI.
            send_tiles(sock, target_ip, resized, q_final, sub_flag, pacing_s=pacing_s)
            # Estimate total bytes for display (approx — each tile ~varies)
            last_frame_bytes = sum(
                len(cv2.imencode('.jpg', resized[TILE_Y[t]:TILE_Y[t]+TILE_H,
                                                 TILE_X[t]:TILE_X[t]+TILE_W],
                                 [int(cv2.IMWRITE_JPEG_QUALITY), q_final])[1])
                for t in range(NUM_TILES))

            # ── Preview window ─────────────────────────
            preview = cv2.resize(resized, (480, 360), interpolation=cv2.INTER_NEAREST)

            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (480, 360), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, 0.85, preview, 0.15, 0)

                # Parse structured diagnostic fields from ESP stats
                fields = {}
                for token in latest_esp_log.split('|'):
                    if ':' in token:
                        k, _, v = token.partition(':')
                        fields[k.strip()] = v.strip()

                # Color-coded diagnostic dashboard
                # Each field is colored by severity so you can spot issues instantly
                def diag_color(key, warn_thresh, err_thresh):
                    try:
                        v = float(fields.get(key, '0').rstrip('%').rstrip('ms').rstrip('KB'))
                        if v >= err_thresh:  return (0,   0, 255)  # red   = error
                        if v >= warn_thresh: return (0, 165, 255)  # orange = warning
                    except: pass
                    return (0, 255, 0)   # green  = ok

                # Parse per-tile stats: T0:ok=N,c=N,to=N
                def tile_stat(t):
                    raw = fields.get(f'T{t}','ok=?,c=?,to=?')
                    kv  = dict(s.split('=') for s in raw.split(',') if '=' in s)
                    ok  = kv.get('ok','?'); c = kv.get('c','?'); to = kv.get('to','?')
                    col = (0,255,0) if c=='0' and to=='0' else (0,165,255) if c=='0' else (0,0,255)
                    return f"T{t}[TL TR BL BR][{t}] ok={ok} corrupt={c} timeout={to}", col

                dashboard = [
                    (f"FPS:    {fields.get('FPS','?'):>6}",           (0,255,0)),
                    (f"Jitter: {fields.get('JIT','?'):>6}",           diag_color('JIT', 10, 30)),
                    (f"DRAM:   {fields.get('DRAM','?'):>6}",          (0,255,0)),
                    (f"PSRAM:  {fields.get('PSRAM','?'):>6}",         (0,255,0)),
                    (f"Pkts:   {fields.get('PKTS','?'):>6}",          (0,255,0)),
                    ("── per tile (ok / corrupt / timeout) ──",       (100,100,100)),
                    (f"TL: {fields.get('T0','?')}",                   diag_color('T0',1,1)),
                    (f"TR: {fields.get('T1','?')}",                   diag_color('T1',1,1)),
                    (f"BL: {fields.get('T2','?')}",                   diag_color('T2',1,1)),
                    (f"BR: {fields.get('T3','?')}",                   diag_color('T3',1,1)),
                ]
                y = 20
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
                    y += 22
            else:
                per_tile = last_frame_bytes // NUM_TILES
                pkts_per = (per_tile + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
                info = (f"Total:{last_frame_bytes}B  PerTile:~{per_tile}B/{pkts_per}pkts "
                        f"Q:{q_final} Sub:{sub} Pace:{pacing_us*100}us "
                        f"{'[Tune]' if current_mode else '[Fast]'}")
                cv2.putText(preview, info, (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
        save_profiles()
        cv2.destroyAllWindows()
        sock.close()


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

    # ── Auto-discover or hardcode ──────────────────────
    ip = quick_find_esp(timeout=5.0)
    # ip = "192.168.x.x"   # ← uncomment and set if auto-discovery fails

    if ip:
        stream_mss_udp(ip, select_display_mss())
    else:
        print("[!] Could not find ESP32. Stream aborted.")

    reset_resolution_timer()