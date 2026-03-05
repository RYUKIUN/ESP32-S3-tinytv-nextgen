import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
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

# UI / trackbar defaults
WINDOW_NAME          = "Stream Control"
UI_W, UI_H           = 480, 560      # cv2 window size
PREVIEW_W, PREVIEW_H = 480, 360      # preview frame size
DEFAULT_FPS          = 30
DEFAULT_QUAL         = 30            # Base Qual trackbar default
DEFAULT_PACING_STEPS = 5             # Pacing trackbar default (5 × 0.2ms = 1ms)
PACING_MAX_STEPS     = 20            # Pacing trackbar max (20 × 0.2ms = 4ms)
PACING_STEP_S        = 0.0004        # seconds per pacing step (0.2ms)

# Encoding / fast-mode heuristics
SHARP_BIAS           = -0.04
DITHER_AMT           = 2             # dither strength when variance is low
DITHER_VAR_THRESH    = 15            # variance below which dithering is applied
SHARP_AMT            = 0.15          # sharpening kernel strength
SHARP_EDGE_THRESH    = 300           # edge density above which sharpening is applied

# Cursor overlay
CURSOR_OUTER_R       = 8             # outer ring radius (px)
CURSOR_INNER_R       = 5             # filled dot radius (px)

# Debug overlay
DEBUG_OVERLAY_ALPHA    = 0.85        # black overlay opacity in debug view
DEBUG_SEND_INTERVAL_S  = 0.5         # seconds between debug-toggle UDP sends

# Diagnostic thresholds (warn, err) — used for colour coding in dashboard
DIAG_FPS_WARN,  DIAG_FPS_ERR   =  15,    10
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,   85
DIAG_JIT_WARN,  DIAG_JIT_ERR   =  10,   30
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  8000, 15000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,    5
DIAG_SRAM_WARN, DIAG_SRAM_ERR  =  50,   20   # free KB (lower = worse)

# Networking / timing
ESP_BEACON_TIMEOUT_S = 5.0           # seconds to wait for ESP32 UDP beacon
SEND_RETRY_SLEEP_S   = 0.0005        # sleep on WSAEWOULDBLOCK before retrying send

# Process priority
UNIX_NICE_LEVEL = -10                # nice value for high-priority on Linux/macOS

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
frame_queue     = Queue(maxsize=1)
settings_lock   = threading.Lock()
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
        else:                p.nice(UNIX_NICE_LEVEL)
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
#  TRANSMIT — chunked UDP
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
                    time.sleep(SEND_RETRY_SLEEP_S)

        # Optional pacing once per tile (not per packet)
        if pacing_s > 0:
            time.sleep(pacing_s)

    return total_bytes

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
        if free < DIAG_SRAM_ERR:  return (0,   0, 255)
        if free < DIAG_SRAM_WARN: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  STREAM + UI
# ─────────────────────────────────────────────
def stream_mss_udp(target_ip: str, monitor_idx: int):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    cv2.createTrackbar("Max FPS",       WINDOW_NAME, DEFAULT_FPS,          60,               lambda x: None)
    cv2.createTrackbar("Base Qual",     WINDOW_NAME, DEFAULT_QUAL,          95,               lambda x: None)
    cv2.createTrackbar("Pacing x0.2ms", WINDOW_NAME, DEFAULT_PACING_STEPS, PACING_MAX_STEPS, lambda x: None)
    cv2.createTrackbar("Debug Info",    WINDOW_NAME, 1,                     1,                lambda x: None)

    threading.Thread(target=capture_worker, args=(monitor_idx,), daemon=True).start()

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
            user_qual    = cv2.getTrackbarPos("Base Qual",       WINDOW_NAME)
            pacing_steps = cv2.getTrackbarPos("Pacing x0.2ms",  WINDOW_NAME)
            debug_state  = cv2.getTrackbarPos("Debug Info",      WINDOW_NAME)
            pacing_s     = pacing_steps * PACING_STEP_S   # each step = 0.2 ms

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
            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                sock.sendto(bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue

            frame = frame_queue.get()

            # ── Draw cursor ─────────────────────────────────────────────────
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), CURSOR_INNER_R, (0,   0, 255),  -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            key, var, edges = get_scene_key(resized)

            # ── Encoding parameters (fast mode always) ────────────────────
            d_amt   = DITHER_AMT if var   < DITHER_VAR_THRESH  else 0
            s_amt   = SHARP_AMT  if edges > SHARP_EDGE_THRESH  else 0.0
            sub     = 420
            q_final = user_qual

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
            preview = cv2.resize(resized, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
            f = latest_esp_stats   # shorthand

            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (PREVIEW_W, PREVIEW_H), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, DEBUG_OVERLAY_ALPHA, preview, 1.0 - DEBUG_OVERLAY_ALPHA, 0)

                # ── Compact diagnostic dashboard ────────────────────────────
                # Fields: FPS, TEMP, JIT, DEC, DROP, SRAM (free/total), PSRAM (free/total)
                dashboard = [
                    (f"FPS  : {f.get('FPS',  '?'):>8}",
                     _diag_color(f.get('FPS', '0'),  DIAG_FPS_WARN,  DIAG_FPS_ERR)),

                    (f"TEMP : {f.get('TEMP', '?'):>7} C",
                     _diag_color(f.get('TEMP','0'),  DIAG_TEMP_WARN, DIAG_TEMP_ERR)),

                    (f"JIT  : {f.get('JIT',  '?'):>7} ms",
                     _diag_color(f.get('JIT', '0'),  DIAG_JIT_WARN,  DIAG_JIT_ERR)),

                    (f"DEC  : {f.get('DEC',  '?'):>7} us",
                     _diag_color(f.get('DEC', '0'),  DIAG_DEC_WARN,  DIAG_DEC_ERR)),

                    (f"DROP : {f.get('DROP', '?'):>8}",
                     _diag_color(f.get('DROP','0'),  DIAG_DROP_WARN, DIAG_DROP_ERR)),

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
                        f"Q:{q_final} Sub:{sub} Pace:{pacing_steps * PACING_STEP_S * 1000:.1f}ms [Fast]")
                cv2.putText(preview, info, (10, PREVIEW_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed  = time.perf_counter() - t_start
            wait_ms  = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
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

    # ── Auto-discover or hardcode ──────────────────────
    ip = quick_find_esp(timeout=ESP_BEACON_TIMEOUT_S)
    # ip = "192.168.x.x"   # ← uncomment and set if auto-discovery fails

    if ip:
        stream_mss_udp(ip, select_display_mss())
    else:
        print("[!] Could not find ESP32. Stream aborted.")

    reset_resolution_timer()