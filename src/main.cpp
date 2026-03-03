/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * PIPELINE ARCHITECTURE
 * ─────────────────────
 *  Two shared slots (back / front) replace the old 4-independent-buffer scheme.
 *
 *  Memory layout (updated):
 *    slot[0].assembly  SRAM  33 KB  ─┐ decoder reads every byte → must be fast
 *    slot[1].assembly  SRAM  33 KB  ─┘
 *    decodeTemp        SRAM  38 KB    Core-1 decode scratch; bswapped in-place
 *    slot[0].fb        PSRAM 38 KB  ─┐ DMA source ONLY — one bulk memcpy from decodeTemp
 *    slot[1].fb        PSRAM 38 KB  ─┘
 *    chunkStorage[4]   PSRAM 134 KB   chunk staging; network writes, not decode-critical
 *
 *  Total SRAM for buffers: ~104 KB  (+38 KB vs previous for decodeTemp)
 *  Key gain: JPEGDEC MCU scatter-writes now hit L1 SRAM cache (was 1200× PSRAM writes).
 *            bswap16_simd runs on SRAM → PIE 128-bit ops fully effective.
 *            PSRAM fb written once via bulk memcpy → sequential, cache-friendly.
 *
 *  Pipeline (steady state):
 *
 *    Core 0 (net)      Core 1 (render)
 *    ─────────────     ───────────────
 *    assemble → slot[back].assembly (SRAM)
 *    post decodeQueue ──────────────→ take decodeQueue
 *    back ^= 1                        decode → decodeTemp (SRAM)
 *    take slotFree[back]              bswap16_simd(decodeTemp)   ← SRAM, full SIMD
 *    assemble → slot[back].assembly   memcpy decodeTemp → slot[s].fb (PSRAM, once)
 *    post decodeQueue  ←──────────── pushImage(slot[s].fb, tileXY)
 *                                     give slotFree[s]
 *    ...
 *
 *  Stats packet (0xAB 0xCD prefix, sent every second when debugEnabled):
 *    FPS:X.X|TEMP:XX.X|JIT:X.X|DEC:XXXX|DROP:X|SRAM:XXXX/XXXX|PSRAM:XXXX/XXXX
 *    DEC  = avg tile decode time in µs (JPEG+bswap+memcpy, NOT pushImage)
 *    DROP = corrupt + timeout count in this 1-second window
 *    SRAM/PSRAM = free_KB/total_KB
 */

 #define LGFX_USE_V1
 #include <LovyanGFX.hpp>
 #include <JPEGDEC.h>
 #include <Arduino.h>
 #include <WiFi.h>
 #include <esp_wifi.h>
 #include <esp_attr.h>
 #include <esp_heap_caps.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include "freertos/semphr.h"
 #include "freertos/queue.h"
 #include <lwip/sockets.h>
 #include <lwip/netdb.h>
 #include <fcntl.h>
 #include <math.h>                  // temperatureRead() needs no extra header in Arduino ESP32
 
 // ─────────────────────────────────────────────
 //  DISPLAY
 // ─────────────────────────────────────────────
 class LGFX : public lgfx::LGFX_Device {
     lgfx::Bus_Parallel8  _bus;
     lgfx::Panel_ILI9341  _panel;
 public:
     LGFX() {
         { auto cfg=_bus.config(); cfg.freq_write=20000000;
           cfg.pin_wr=1; cfg.pin_rd=40; cfg.pin_rs=2;
           cfg.pin_d0=5; cfg.pin_d1=4;  cfg.pin_d2=10;
           cfg.pin_d3=9; cfg.pin_d4=3;  cfg.pin_d5=8;
           cfg.pin_d6=7; cfg.pin_d7=6;
           _bus.config(cfg); _panel.setBus(&_bus); }
         { auto cfg=_panel.config();
           cfg.pin_cs=41; cfg.pin_rst=39; cfg.pin_busy=-1;
           cfg.panel_width=240; cfg.panel_height=320;
           cfg.offset_x=0; cfg.offset_y=0; cfg.offset_rotation=0;
           cfg.dummy_read_pixel=8;
           cfg.readable=false; cfg.invert=false;
           cfg.rgb_order=false; cfg.dlen_16bit=false; cfg.bus_shared=false;
           _panel.config(cfg); }
         setPanel(&_panel);
     }
 };
 static LGFX lcd;
 
 // ─────────────────────────────────────────────
 //  CONFIG
 // ─────────────────────────────────────────────
 const char* WIFI_SSID  = "streaming";
 const char* WIFI_PASS  = "12345678";
 const int   UDP_PORT   = 12345;
 
 #define SCREEN_W         320
 #define SCREEN_H         240
 #define NUM_TILES        4
 #define TILE_W           160
 #define TILE_H           120
 #define TILE_PIXELS      (TILE_W * TILE_H)          // 19 200
 #define CHUNK_DATA_SIZE  1400
 #define MAX_TILE_CHUNKS  24          // 24 × 1400 = 33.6 KB max JPEG per tile
 #define MAX_TILE_JPEG    (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)
 #define TILE_TIMEOUT_MS  200
 
 // Screen position of each tile:  TL TR BL BR
 static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
 static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };
 
 // ─────────────────────────────────────────────
 //  PIPELINE SLOTS  (2 shared decode/display buffers)
 // ─────────────────────────────────────────────
 struct PipeSlot {
     uint8_t*  assembly;   // SRAM — JPEGDEC reads here; single-cycle access critical
 };
 static PipeSlot slot[2];
 
 // SRAM scratch for Core-1 decode.  JPEGDEC scatter-writes go here (not PSRAM).
 // bswap16_simd also runs here.  One bulk memcpy then fills slot[s].fb in PSRAM.
 // Core-1 exclusive — no synchronisation needed.
 static uint16_t* decodeTemp = nullptr;   // allocated in setup(), MALLOC_CAP_INTERNAL
 
// Final per-tile framebuffer in PSRAM (one per screen tile).
// We decode tiles into these buffers, then present a whole frame (all 4 tiles)
// at once to eliminate visible inter-tile temporal "ripple".
static uint16_t* tileFb[NUM_TILES] = { nullptr, nullptr, nullptr, nullptr };

 // Message passed through the decode queue
 struct DecodeMsg {
    uint8_t  frameId;  // frame sequence (0-255) for frame-sync presentation
     uint8_t  tId;     // which tile position (0-3) → determines screen XY
     uint8_t  slotIdx; // which PipeSlot holds the assembled JPEG
     uint16_t len;     // JPEG byte count in slot[slotIdx].assembly
 };
 
 // Pipeline synchronisation
 static QueueHandle_t     decodeQueue;    // depth-1 queue: net → renderer
 static SemaphoreHandle_t slotFree[2];   // given when renderer finishes slot
 
 // ─────────────────────────────────────────────
 //  CHUNK REASSEMBLY STATE  (one per tile position)
 // ─────────────────────────────────────────────
 struct TileState {
     uint8_t* chunkBuf[MAX_TILE_CHUNKS]; // → PSRAM chunkStorage slab
     uint16_t chunkLen[MAX_TILE_CHUNKS];
     bool     chunkGot[MAX_TILE_CHUNKS];
     uint8_t  frameId      = 0xFF;
     uint8_t  totalChunks  = 0;
     uint16_t frameSize    = 0;
     uint8_t  chunksGot    = 0;
     uint32_t firstChunkMs = 0;
     // Stats — written by Core 0, reset by Core 0 after each stat window
     uint32_t stat_decoded = 0;
     uint32_t stat_corrupt = 0;
     uint32_t stat_timeout = 0;
 };
 static TileState tiles[NUM_TILES];
 static uint8_t*  tileChunkStorage[NUM_TILES] = {};
 
 // ─────────────────────────────────────────────
 //  CROSS-CORE STATS  (Core 1 writes, Core 0 reads for UDP report)
 //  32-bit aligned → single-instruction read/write on LX7, no tearing.
 // ─────────────────────────────────────────────
 static volatile uint32_t g_avgDecodeUs = 0;  // avg tile decode µs (excl. pushImage)
 
 // ─────────────────────────────────────────────
 //  GLOBAL STATE
 // ─────────────────────────────────────────────
 static bool     debugEnabled      = false;
 static char     debugBuf[256];             // 256 B is plenty for the new compact format
 static int      g_sock            = -1;
 static struct   sockaddr_in g_remoteAddr;
 static bool     g_remoteAddrValid = false;
 static float    stat_jitter       = 0.0f;  // Core 1 writes, Core 0 reads — 32-bit float, OK
 static uint32_t stat_prevMs       = 0;
 
 // ─────────────────────────────────────────────
 //  TILE HELPERS
 // ─────────────────────────────────────────────
 static IRAM_ATTR void resetTile(uint8_t t) {
     memset(tiles[t].chunkGot, 0, sizeof(tiles[t].chunkGot));
     tiles[t].frameId      = 0xFF;
     tiles[t].totalChunks  = 0;
     tiles[t].frameSize    = 0;
     tiles[t].chunksGot    = 0;
     tiles[t].firstChunkMs = 0;
 }
 
 // Assemble complete tile JPEG from chunks into dst (slot[].assembly, SRAM).
 // Returns assembled byte count, or 0 on corruption.
 static IRAM_ATTR int assembleTileInto(uint8_t t, uint8_t* dst) {
     TileState& ts = tiles[t];
     if (ts.totalChunks == 0) return 0;
     int offset = 0;
     for (uint8_t c = 0; c < ts.totalChunks; c++) {
         if (!ts.chunkGot[c]) return 0;
         memcpy(dst + offset, ts.chunkBuf[c], ts.chunkLen[c]);
         offset += ts.chunkLen[c];
     }
     // Validate JPEG SOI / EOI markers
     if (offset < 4 ||
         dst[0] != 0xFF || dst[1] != 0xD8 ||
         dst[offset-2] != 0xFF || dst[offset-1] != 0xD9) {
         Serial.printf("[TILE%u] bad markers len=%d [%02X%02X..%02X%02X]\n",
             t, offset, dst[0], dst[1], dst[offset-2], dst[offset-1]);
         ts.stat_corrupt++;
         return 0;
     }
     return offset;
 }
 
 // ─────────────────────────────────────────────
 //  SIMD + JPEGDEC PIPELINE
 // ─────────────────────────────────────────────
 //
 // Step 1 — JPEGDEC fires mcuCallback for each 16×16 MCU block.
 //           Pixels scatter-written into decodeTemp (SRAM, not PSRAM).
 //           RGB565 little-endian (JPEGDEC native).
 //           -DESP32S3 enables JPEGDEC PIE SIMD for IDCT and YCbCr→RGB565.
 //
 // Step 2 — bswap16_simd() runs on decodeTemp (SRAM → L1 cache).
 //           GCC -O3 vectorizes uint32_t mask/shift into PIE EE.xxx 128-bit ops:
 //           8 pixels/cycle on LX7.  Effective because source is in L1, not PSRAM.
 //
 // Step 3 — One bulk memcpy: decodeTemp (SRAM) → slot[s].fb (PSRAM).
 //           Sequential write, cache-line aligned, much cheaper than 1200× scatter.
 //
 // Step 4 — lcd.pushImage() sends slot[s].fb to display.
 //           1 address-window setup per tile.
 
 static JPEGDEC jpeg_dec;
 
 struct McuCtx { uint16_t* fb; };
 static McuCtx mcuCtx;
 
 // MCU callback: scatter-write into decodeTemp (SRAM).
 static IRAM_ATTR int mcuCallback(JPEGDRAW* pDraw) {
     uint16_t*       dst = ((McuCtx*)pDraw->pUser)->fb + pDraw->y * TILE_W + pDraw->x;
     const uint16_t* src = (const uint16_t*)pDraw->pPixels;
     int w = pDraw->iWidth, h = pDraw->iHeight;
     for (int r = 0; r < h; r++)
         memcpy(dst + r * TILE_W, src + r * w, (size_t)w * 2);
     return 1;
 }
 
 // SIMD byte-swap: LE RGB565 → BE for ILI9341.
 // Processes 2 pixels per uint32_t word.
 // GCC -O3 on LX7 vectorizes into PIE 128-bit ops (8 pixels/cycle).
 static IRAM_ATTR __attribute__((optimize("O3")))
 void bswap16_simd(uint16_t* buf, int n) {
     uint32_t* p = (uint32_t*)buf;
     int w = n >> 1;
     for (int i = 0; i < w; i++) {
         uint32_t v = p[i];
         p[i] = ((v & 0xFF00FF00u) >> 8u) | ((v & 0x00FF00FFu) << 8u);
     }
     if (n & 1) buf[n-1] = __builtin_bswap16(buf[n-1]);
 }
 
 // Full decode pipeline for one slot.
 // Returns true on success; on false the slot semaphore must be given back by caller.
// decodeUs (output): time spent on decode+bswap+memcpy in µs (not LCD push).
 static IRAM_ATTR bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
     PipeSlot& s = slot[msg.slotIdx];
 
     if ((uintptr_t)s.assembly & 3) {
         Serial.printf("[DEC] slot%u assembly unaligned\n", msg.slotIdx);
         decodeUs = 0;
         return false;
     }
 
     // ── Decode into SRAM scratch (avoids 1200× PSRAM scatter-writes) ──────
     mcuCtx.fb = decodeTemp;
     if (!jpeg_dec.openRAM(s.assembly, msg.len, mcuCallback)) {
         Serial.printf("[DEC] slot%u open err=%d\n", msg.slotIdx, jpeg_dec.getLastError());
         decodeUs = 0;
         return false;
     }
     jpeg_dec.setPixelType(RGB565_LITTLE_ENDIAN);
     jpeg_dec.setUserPointer(&mcuCtx);
 
     uint32_t t0 = micros();
     int rc = jpeg_dec.decode(0, 0, 0);
     jpeg_dec.close();
 
     if (!rc) {
         Serial.printf("[DEC] slot%u decode err=%d len=%u\n",
                       msg.slotIdx, jpeg_dec.getLastError(), msg.len);
         decodeUs = 0;
         return false;
     }
 
     // ── SIMD bswap on SRAM — full PIE vector throughput ───────────────────
     bswap16_simd(decodeTemp, TILE_PIXELS);
 
    // ── Bulk copy SRAM→PSRAM — sequential, one cache-line stream ──────────
    // Copy into the final per-tile buffer for later frame-synchronised present.
    if (msg.tId >= NUM_TILES || tileFb[msg.tId] == nullptr) {
        decodeUs = 0;
        return false;
    }
    memcpy(tileFb[msg.tId], decodeTemp, TILE_PIXELS * 2);
 
     decodeUs = micros() - t0;   // decode + bswap + memcpy, not pushImage
 
     return true;
 }
 
 // ─────────────────────────────────────────────
 //  NETWORK TASK  (Core 0)
 // ─────────────────────────────────────────────
 // Packet format:
 //   Data:    [0xAA 0xBB frameId tileId chunkId totalChunks sizeHi sizeLo] + payload
 //   Control: [0xAA 0xCC 0x01 debugState]
 //
 // Pipeline flow when tile completes:
 //   1. xSemaphoreTake(slotFree[back])     — wait for renderer to vacate slot
 //   2. assembleTileInto(tId, slot[back])  — PSRAM chunks → SRAM assembly
 //   3. xQueueSend(decodeQueue, &msg)      — block until renderer is ready
 //   4. back ^= 1
 static IRAM_ATTR void networkTask(void*) {
     g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
     if (g_sock < 0) { Serial.println("[NET] socket fail"); vTaskDelete(NULL); return; }
 
     int rcvbuf = 65536;
     setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
 
     struct sockaddr_in local = {};
     local.sin_family = AF_INET;
     local.sin_port   = htons(UDP_PORT);
     local.sin_addr.s_addr = INADDR_ANY;
     if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
         Serial.println("[NET] bind fail"); close(g_sock); vTaskDelete(NULL); return;
     }
     fcntl(g_sock, F_SETFL, O_NONBLOCK);
     Serial.printf("[NET] UDP ready port=%d\n", UDP_PORT);
 
     // rxBuf is static — avoids stack pressure (stack budgeted at 10 KB)
     static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
     struct sockaddr_in sender;
     socklen_t slen = sizeof(sender);
     uint32_t  lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0, pktCount = 0;
     uint8_t   back = 0;
 
     while (true) {
         int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                          (struct sockaddr*)&sender, &slen);
 
         if (n < 0) {
             // 1 ms select — responsive first-packet detection, no wasted 5 ms sleeps
             fd_set rfds; FD_ZERO(&rfds); FD_SET(g_sock, &rfds);
             struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
             select(g_sock + 1, &rfds, NULL, NULL, &tv);
 
             // Beacon if no traffic for 2 s
             if ((millis()-lastBeaconMs) > 2000 && (millis()-lastPktMs) > 2000) {
                 struct sockaddr_in bc = {};
                 bc.sin_family = AF_INET;
                 bc.sin_port   = htons(UDP_PORT);
                 bc.sin_addr.s_addr = htonl(INADDR_BROADCAST);
                 int so = 1;
                 setsockopt(g_sock, SOL_SOCKET, SO_BROADCAST, &so, sizeof(so));
                 const char* b = "ESP32_READY";
                 sendto(g_sock, b, strlen(b), 0, (struct sockaddr*)&bc, sizeof(bc));
                 lastBeaconMs = millis();
             }
             continue;
         }
 
         lastPktMs = millis(); pktCount++;
         if (n < 4 || rxBuf[0] != 0xAA) { portYIELD(); continue; }
         memcpy(&g_remoteAddr, &sender, sizeof(sender));
         g_remoteAddrValid = true;
 
         // ── Control packet ────────────────────────────────────────────────
         if (rxBuf[1] == 0xCC) {
             if (n >= 4 && rxBuf[2] == 0x01) debugEnabled = (rxBuf[3] == 1);
             portYIELD(); continue;
         }
 
         // ── Tile data chunk ───────────────────────────────────────────────
         if (rxBuf[1] != 0xBB || n < 8) { portYIELD(); continue; }
         uint8_t  fId     = rxBuf[2];
         uint8_t  tId     = rxBuf[3];
         uint8_t  cId     = rxBuf[4];
         uint8_t  nChunks = rxBuf[5];
         uint16_t fSize   = ((uint16_t)rxBuf[6] << 8) | rxBuf[7];
         int      dataLen = n - 8;
         if (tId >= NUM_TILES || dataLen <= 0) { portYIELD(); continue; }
 
         TileState& ts = tiles[tId];
 
         // Timeout stale reassembly
         if (ts.firstChunkMs > 0 && (millis() - ts.firstChunkMs) > TILE_TIMEOUT_MS) {
             Serial.printf("[TILE%u] timeout got=%u/%u\n", tId, ts.chunksGot, ts.totalChunks);
             ts.stat_timeout++;
             resetTile(tId);
         }
 
         // New frame for this tile
         if (fId != ts.frameId) {
             resetTile(tId);
             ts.frameId     = fId;
             ts.totalChunks = nChunks;
             ts.frameSize   = fSize;
             ts.firstChunkMs = millis();
         }
 
         // Store chunk into PSRAM staging area
         if (cId < MAX_TILE_CHUNKS && !ts.chunkGot[cId]) {
             memcpy(ts.chunkBuf[cId], &rxBuf[8], dataLen);
             ts.chunkLen[cId] = (uint16_t)dataLen;
             ts.chunkGot[cId] = true;
             ts.chunksGot++;
         }
 
         // All chunks received → feed the pipeline
         if (ts.chunksGot >= ts.totalChunks) {
 
             // Wait for renderer to vacate the slot we're about to fill.
             // In steady state returns immediately — renderer finished while
             // we were receiving this tile's remaining chunks.
             xSemaphoreTake(slotFree[back], portMAX_DELAY);
 
             int len = assembleTileInto(tId, slot[back].assembly);
 
             if (len > 0) {
                DecodeMsg msg = { fId, tId, back, (uint16_t)len };
                 xQueueSend(decodeQueue, &msg, portMAX_DELAY);
                 back ^= 1;
             } else {
                 // Corrupt JPEG: slot was never used — return semaphore immediately
                 xSemaphoreGive(slotFree[back]);
                 // stat_corrupt already incremented inside assembleTileInto
             }
 
             resetTile(tId);
         }
 
         // ── Periodic stat report ─────────────────────────────────────────
         if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > 1000) {
             uint32_t el = millis() - lastStatMs;
 
             // FPS: sum decoded tiles / 4 tiles per frame / elapsed seconds
             uint32_t totalDec = 0;
             uint32_t totalDrop = 0;
             for (int i = 0; i < NUM_TILES; i++) {
                 totalDec  += tiles[i].stat_decoded;
                 totalDrop += tiles[i].stat_corrupt + tiles[i].stat_timeout;
             }
             float fps = (totalDec / 4.0f) / (el / 1000.0f);
 
             // Memory
             uint32_t freeSRAM  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
             uint32_t totalSRAM = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
             uint32_t freePSR   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
             uint32_t totalPSR  = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
 
             // Temperature (Arduino ESP32 built-in, no extra driver needed)
             float tempC = temperatureRead();
 
             // Decode time from Core 1 (volatile read — 32-bit aligned, atomic on LX7)
             uint32_t decUs = g_avgDecodeUs;
 
             snprintf(debugBuf, sizeof(debugBuf),
                 "%c%cFPS:%.1f|TEMP:%.1f|JIT:%.1f|DEC:%lu|DROP:%lu"
                 "|SRAM:%lu/%lu|PSRAM:%lu/%lu",
                 0xAB, 0xCD,
                 fps, tempC, stat_jitter,
                 decUs, totalDrop,
                 freeSRAM / 1024, totalSRAM / 1024,
                 freePSR  / 1024, totalPSR  / 1024);
 
             sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                    (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));
 
             // Reset per-window counters
             for (int i = 0; i < NUM_TILES; i++)
                 tiles[i].stat_decoded = tiles[i].stat_corrupt = tiles[i].stat_timeout = 0;
             pktCount = 0;
             lastStatMs = millis();
         }
 
         portYIELD();
     }
 }
 
 // ─────────────────────────────────────────────
 //  DISPLAY HELPERS
 // ─────────────────────────────────────────────
 static void statusLine(uint8_t row, const char* label, const char* value,
                        uint32_t col = TFT_WHITE) {
     int y = 58 + row * 22;
     lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
     lcd.setTextColor(0x7BEF, TFT_BLACK); lcd.drawString(label, 8, y + 3);
     lcd.setTextColor(col,    TFT_BLACK); lcd.drawString(value, 138, y + 3);
 }
 
 static void drawBootHeader() {
     lcd.fillScreen(TFT_BLACK);
     lcd.setTextFont(2); lcd.setTextSize(1);
     lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
     lcd.setTextColor(TFT_CYAN, 0x1082); lcd.setTextSize(2);
     lcd.drawString("ESP32-S3 STREAM", 8, 6);
     lcd.setTextSize(1); lcd.setTextColor(0x7BEF, 0x1082);
     lcd.drawString("ILI9341  320x240  ping-pong", 8, 34);
     lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
 }
 
 // ─────────────────────────────────────────────
 //  SETUP
 // ─────────────────────────────────────────────
 void setup() {
     Serial.begin(115200);
     uint32_t t0 = millis();
     while (!Serial && (millis() - t0) < 2000) delay(10);
     Serial.println("\n[BOOT] ping-pong pipeline v2 (SRAM decode)");
 
     lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
     lcd.setTextFont(2); lcd.setTextSize(1);
     drawBootHeader();
     statusLine(0, "Display:", "OK", TFT_GREEN);
 
     bool psramOk = psramFound();
     statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
     if (!psramOk) { while (1) delay(1000); }
 
     // ── Allocate SRAM decode scratch (Core-1 exclusive) ──────────────────
     // 38 KB: JPEGDEC scatter-writes + bswap run here before bulk copy to PSRAM.
     decodeTemp = (uint16_t*)heap_caps_aligned_alloc(
         16, TILE_PIXELS * 2,
         MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
     if (!decodeTemp) {
         Serial.println("[ERROR] decodeTemp SRAM alloc failed");
         statusLine(2, "DecTemp:", "ALLOC FAILED!", TFT_RED);
         while (1) delay(1000);
     }
 
     // ── Allocate pipeline slots ───────────────────────────────────────────
     bool allocOk = true;
     for (int s = 0; s < 2; s++) {
         slot[s].assembly = (uint8_t*)heap_caps_malloc(
             MAX_TILE_JPEG, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
         if (!slot[s].assembly) {
             Serial.printf("[ERROR] slot[%d].assembly SRAM alloc failed\n", s);
             allocOk = false; break;
         }
     }
 
    // ── Allocate final per-tile framebuffers in PSRAM ─────────────────────
    for (int t = 0; t < NUM_TILES && allocOk; t++) {
        tileFb[t] = (uint16_t*)heap_caps_aligned_alloc(
            16, TILE_PIXELS * 2, MALLOC_CAP_SPIRAM);
        if (!tileFb[t]) {
            Serial.printf("[ERROR] tileFb[%d] PSRAM alloc failed\n", t);
            allocOk = false; break;
        }
    }

     // ── Allocate chunk staging → PSRAM ───────────────────────────────────
     for (int t = 0; t < NUM_TILES && allocOk; t++) {
         tileChunkStorage[t] = (uint8_t*)heap_caps_malloc(
             (size_t)MAX_TILE_CHUNKS * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
         if (!tileChunkStorage[t]) {
             Serial.printf("[ERROR] tile[%d] chunkStorage PSRAM alloc failed\n", t);
             allocOk = false; break;
         }
         for (int c = 0; c < MAX_TILE_CHUNKS; c++)
             tiles[t].chunkBuf[c] = tileChunkStorage[t] + (size_t)c * CHUNK_DATA_SIZE;
     }
 
     if (!allocOk) {
         statusLine(2, "Buffers:", "ALLOC FAILED!", TFT_RED);
         while (1) delay(1000);
     }
 
     // ── Pipeline sync primitives ──────────────────────────────────────────
     decodeQueue = xQueueCreate(1, sizeof(DecodeMsg));
     for (int s = 0; s < 2; s++) {
         slotFree[s] = xSemaphoreCreateBinary();
         xSemaphoreGive(slotFree[s]);
     }
 
     // Print memory layout
     Serial.printf("[MEM] decodeTemp       : %u B SRAM (16-byte aligned)\n", TILE_PIXELS*2);
     Serial.printf("[MEM] slot[0].assembly : %u B SRAM\n", MAX_TILE_JPEG);
     Serial.printf("[MEM] slot[1].assembly : %u B SRAM\n", MAX_TILE_JPEG);
    Serial.printf("[MEM] tileFb x4        : %u B PSRAM (16-byte aligned)\n", NUM_TILES*TILE_PIXELS*2);
     Serial.printf("[MEM] chunkStorage x4  : %u B PSRAM\n", NUM_TILES*MAX_TILE_CHUNKS*CHUNK_DATA_SIZE);
     Serial.printf("[MEM] free SRAM  : %lu KB / %lu KB\n",
         heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024,
         heap_caps_get_total_size(MALLOC_CAP_INTERNAL)/1024);
     Serial.printf("[MEM] free PSRAM : %lu KB / %lu KB\n",
         heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024,
         heap_caps_get_total_size(MALLOC_CAP_SPIRAM)/1024);
 
     statusLine(2, "Buffers:", "2-slot SRAM-dec", TFT_GREEN);
 
     // ── WiFi ──────────────────────────────────────────────────────────────
     statusLine(3, "WiFi:", "Connecting...", TFT_YELLOW);
     WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.begin(WIFI_SSID, WIFI_PASS);
     uint32_t ws = millis(); uint8_t tick = 0;
     while (WiFi.status() != WL_CONNECTED) {
         delay(250); tick++;
         char buf[24]; snprintf(buf, sizeof(buf), "Conn%.*s", tick % 5, ".....");
         statusLine(3, "WiFi:", buf, TFT_YELLOW);
         if (millis() - ws > 20000) {
             statusLine(3, "WiFi:", "TIMEOUT!", TFT_RED);
             delay(3000); ESP.restart();
         }
     }
     esp_wifi_set_ps(WIFI_PS_NONE);
     String ip = WiFi.localIP().toString();
     char ipBuf[36]; snprintf(ipBuf, sizeof(ipBuf), "%s (%ddBm)", ip.c_str(), WiFi.RSSI());
     statusLine(3, "WiFi:", ipBuf, TFT_GREEN);
     statusLine(4, "UDP:", String(UDP_PORT).c_str(), TFT_CYAN);
     statusLine(5, "Mode:", "4-tile ping-pong", TFT_CYAN);
     statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
     Serial.printf("[OK] WiFi: %s\n", ip.c_str());
 
     // 10 KB stack — headroom for debug snprintf + lwIP frames; rxBuf is static
     xTaskCreatePinnedToCore(networkTask, "NetTask", 10240, NULL, 2, NULL, 0);
     Serial.println("[OK] Ready.");
 }
 
 // ─────────────────────────────────────────────
 //  MAIN LOOP  (Core 1 — renderer)
 // ─────────────────────────────────────────────
 // Blocks on decodeQueue. When net delivers a DecodeMsg:
 //   decode → decodeTemp (SRAM)
 //   bswap → decodeTemp
 //   memcpy → slot[s].fb (PSRAM)
 //   pushImage → display
 //   give slotFree[s]
 //
 // g_avgDecodeUs updated every 16 tiles (volatile write → Core 0 reads for stats).
 void loop() {
     static bool     streamStarted = false;
     static uint32_t decodeAcc     = 0;   // accumulated decode µs
     static uint32_t decodeCount   = 0;

    // Frame-sync presentation state (Core 1 only)
    static uint8_t  pendingFrame  = 0xFF;
    static uint8_t  readyMask     = 0;
    static uint32_t frameStartMs  = 0;
 
     DecodeMsg msg;
     // 40 ms timeout → ~25 fps floor before logging idle
     if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;
 
    // If we started a frame but it never completes, discard it to avoid "stuck" output.
    // (Should be rare when DROP is ~0.)
    if (pendingFrame != 0xFF && frameStartMs > 0 && (millis() - frameStartMs) > 150) {
        pendingFrame = 0xFF;
        readyMask    = 0;
        frameStartMs = 0;
    }

    // New frameId => start collecting tiles for that frame.
    if (pendingFrame == 0xFF || msg.frameId != pendingFrame) {
        pendingFrame = msg.frameId;
        readyMask    = 0;
        frameStartMs = millis();
    }

     uint32_t decUs = 0;
     bool ok = decodeSlot(msg, decUs);
 
     // Release slot immediately — net can now write to it
     xSemaphoreGive(slotFree[msg.slotIdx]);
 
     if (ok) {
         tiles[msg.tId].stat_decoded++;
         decodeAcc   += decUs;
         decodeCount++;

        readyMask |= (uint8_t)(1u << msg.tId);
 
         // Update cross-core average every 16 tiles — amortises volatile write cost
         if (decodeCount >= 16) {
             g_avgDecodeUs = decodeAcc / decodeCount;
             decodeAcc = 0; decodeCount = 0;
         }
 
         if (!streamStarted) {
             streamStarted = true;
             statusLine(6, "Status:", "STREAMING!", TFT_GREEN);
             Serial.printf("[RENDER] first tile=%u slot=%u len=%u dec=%luus\n",
                           msg.tId, msg.slotIdx, msg.len, decUs);
             delay(200);
         }
 
         if ((tiles[msg.tId].stat_decoded % 120) == 0 && g_avgDecodeUs > 0) {
             Serial.printf("[RENDER] avg decode: %lu us  (%lu fps-equiv)\n",
                           g_avgDecodeUs, 1000000ul / g_avgDecodeUs);
         }

        // Present only when all 4 tiles of the same frame are ready.
        // This eliminates inter-tile temporal skew ("ripple"/displacement).
        if (readyMask == 0x0F) {
            for (int t = 0; t < NUM_TILES; t++) {
                lcd.pushImage(TILE_X[t], TILE_Y[t], TILE_W, TILE_H, tileFb[t]);
            }
            readyMask    = 0;
            pendingFrame = 0xFF;
            frameStartMs = 0;
        }
     }
 
     // Jitter — measured at tile-delivery rate (millis, Core 1 only)
     uint32_t now = millis();
     if (stat_prevMs > 0) {
         static uint32_t lastIv = 0;
         uint32_t iv = now - stat_prevMs;
         if (lastIv > 0) {
             int32_t d = (int32_t)iv - (int32_t)lastIv;
             stat_jitter += (fabsf((float)d) - stat_jitter) / 16.0f;
         }
         lastIv = iv;
     }
     stat_prevMs = now;
 }