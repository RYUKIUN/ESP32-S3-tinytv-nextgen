/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320×240
 *
 * Key changes vs old version:
 *  - TFT_eSPI  →  LovyanGFX  (native i80 parallel bus support)
 *  - 160×128   →  320×240
 *  - XOR-based FEC recovery for dropped UDP chunks
 *  - PSRAM used for sprites & chunk reassembly buffers
 */

#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <JPEGDEC.h>
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
// Raw lwIP sockets — gives us control over SO_RCVBUF and avoids WiFiUDP wrapper overhead
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>

// ─────────────────────────────────────────────
//  DISPLAY CONFIGURATION  (matches pref_for_tft)
// ─────────────────────────────────────────────
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_Parallel8  _bus;
    lgfx::Panel_ILI9341  _panel;

public:
    LGFX() {
        // ── 8-bit i80 bus ──
        {
            auto cfg        = _bus.config();
            cfg.freq_write  = 20000000;   // 20 MHz – safe & ~95 FPS raw
            cfg.pin_wr      = 1;
            cfg.pin_rd      = 40;
            cfg.pin_rs      = 2;          // D/C line
            cfg.pin_d0      = 5;
            cfg.pin_d1      = 4;
            cfg.pin_d2      = 10;
            cfg.pin_d3      = 9;
            cfg.pin_d4      = 3;
            cfg.pin_d5      = 8;
            cfg.pin_d6      = 7;
            cfg.pin_d7      = 6;
            _bus.config(cfg);
            _panel.setBus(&_bus);
        }

        // ── ILI9341 panel ──
        {
            auto cfg         = _panel.config();
            cfg.pin_cs       = 41;
            cfg.pin_rst      = 39;
            cfg.pin_busy     = -1;
            cfg.panel_width  = 240;   // physical dimensions
            cfg.panel_height = 320;
            cfg.offset_x     = 0;
            cfg.offset_y     = 0;
            cfg.offset_rotation = 0;
            cfg.dummy_read_pixel = 8;
            cfg.readable     = false;
            cfg.invert       = false;
            cfg.rgb_order    = false;
            cfg.dlen_16bit   = false;
            cfg.bus_shared   = false;
            _panel.config(cfg);
        }

        setPanel(&_panel);
    }
};

static LGFX lcd;
static LGFX_Sprite* spr[2];

// ─────────────────────────────────────────────
//  NETWORK CONFIG
// ─────────────────────────────────────────────
const char* WIFI_SSID  = "streaming";
const char* WIFI_PASS  = "12345678";
const int   UDP_PORT   = 12345;

// ─────────────────────────────────────────────
//  FEC / CHUNK CONSTANTS  (must match capture.py)
// ─────────────────────────────────────────────
#define CHUNK_DATA_SIZE   1400
#define FEC_GROUP_SIZE    4    // was 8; smaller groups = more parity overhead (25%)
                               // but each parity covers fewer chunks → better loss recovery.
                               // Must match FEC_GROUP_SIZE in capture.py.
#define MAX_CHUNKS        64                          // 64×1400 = 89.6 KB max frame
#define MAX_PARITY        ((MAX_CHUNKS + FEC_GROUP_SIZE - 1) / FEC_GROUP_SIZE)  // 16 with group=4
#define MAX_JPEG_SIZE     (MAX_CHUNKS * CHUNK_DATA_SIZE)   // ~90 KB
#define FRAME_TIMEOUT_MS  150       // slightly more slack for paced sender
#define SCREEN_W          320
#define SCREEN_H          240

// ─────────────────────────────────────────────
//  FRAME REASSEMBLY STATE  (all in PSRAM)
// ─────────────────────────────────────────────
static uint8_t* chunkBuf[MAX_CHUNKS];     // pointers into one big PSRAM block
static uint16_t chunkLen[MAX_CHUNKS];
static bool     chunkGot[MAX_CHUNKS];

static uint8_t* parityBuf[MAX_PARITY];
static bool     parityGot[MAX_PARITY];

static uint8_t  curFrameId        = 0xFF;
static uint8_t  totalDataChunks   = 0;
static uint8_t  totalParityChunks = 0;
static uint16_t totalFrameSize    = 0;     // original JPEG byte count
static uint32_t firstChunkMs      = 0;
static uint8_t  chunksGotCount    = 0;

// Double-buffer for decoded frames
static uint8_t* jpegAssembly  = nullptr;  // PSRAM
static uint8_t* jpegReadyBuf  = nullptr;  // PSRAM – stable copy passed to renderer
static int      jpegReadyLen  = 0;
static bool     jpegReadyFlag = false;

// Big PSRAM slab for all chunk data
static uint8_t* chunkStorage  = nullptr;  // MAX_CHUNKS × CHUNK_DATA_SIZE
static uint8_t* parityStorage = nullptr;  // MAX_PARITY × CHUNK_DATA_SIZE

SemaphoreHandle_t frameSem;
SemaphoreHandle_t jpegMutex;

// ─────────────────────────────────────────────
//  STATS
// ─────────────────────────────────────────────
static uint32_t stat_framesReceived = 0;
static uint32_t stat_framesDropped  = 0;
static uint32_t stat_lastReport     = 0;
static float    stat_jitter         = 0.0f;
static uint32_t stat_prevFrameMs    = 0;

static bool       debugEnabled = false;
static char       debugBuf[300];

// ─────────────────────────────────────────────
//  FEC: XOR parity recovery
// ─────────────────────────────────────────────
static void resetFrameState() {
    memset(chunkGot, 0, sizeof(chunkGot));
    memset(parityGot, 0, sizeof(parityGot));
    totalDataChunks   = 0;
    totalParityChunks = 0;
    totalFrameSize    = 0;
    chunksGotCount    = 0;
    firstChunkMs      = 0;
}

/*
 * Try to recover any single missing chunk per FEC group using XOR parity.
 * Returns true if the frame can now be assembled (all chunks present).
 */
static bool attemptFecRecovery() {
    if (totalDataChunks == 0) return false;

    uint8_t numGroups = (totalDataChunks + FEC_GROUP_SIZE - 1) / FEC_GROUP_SIZE;

    for (uint8_t g = 0; g < numGroups; g++) {
        uint8_t gStart = g * FEC_GROUP_SIZE;
        uint8_t gEnd   = min((uint8_t)(gStart + FEC_GROUP_SIZE), totalDataChunks);
        // Count missing in this group
        int8_t missingIdx = -1;
        uint8_t missingCount = 0;
        for (uint8_t c = gStart; c < gEnd; c++) {
            if (!chunkGot[c]) { missingIdx = c; missingCount++; }
        }

        if (missingCount == 0) continue;

        if (missingCount == 1 && parityGot[g]) {
            // Recover: recovered = parity XOR all_other_chunks
            uint8_t* recovered = chunkBuf[missingIdx];
            memcpy(recovered, parityBuf[g], CHUNK_DATA_SIZE);

            for (uint8_t c = gStart; c < gEnd; c++) {
                if (c == (uint8_t)missingIdx) continue;
                for (int b = 0; b < CHUNK_DATA_SIZE; b++) {
                    recovered[b] ^= chunkBuf[c][b];
                }
            }

            // Determine recovered chunk length
            if (missingIdx == totalDataChunks - 1) {
                // Last chunk – use remainder
                chunkLen[missingIdx] = totalFrameSize - (uint16_t)(missingIdx) * CHUNK_DATA_SIZE;
            } else {
                chunkLen[missingIdx] = CHUNK_DATA_SIZE;
            }
            chunkGot[missingIdx] = true;
            chunksGotCount++;
        }
        // If missingCount > 1 and no parity, we can't recover – frame will be incomplete
    }

    // Check if all chunks present now
    for (uint8_t c = 0; c < totalDataChunks; c++) {
        if (!chunkGot[c]) return false;
    }
    return true;
}

/*
 * Assemble JPEG from received chunks into jpegAssembly buffer.
 * Returns assembled size, or 0 on failure.
 */
static int assembleJpeg() {
    if (!jpegAssembly || totalDataChunks == 0) return 0;

    int offset = 0;
    for (uint8_t c = 0; c < totalDataChunks; c++) {
        if (!chunkGot[c]) return 0;
        memcpy(jpegAssembly + offset, chunkBuf[c], chunkLen[c]);
        offset += chunkLen[c];
    }
    return offset;
}

// ─────────────────────────────────────────────
//  NETWORK TASK  (Core 0)  — raw lwIP socket
// ─────────────────────────────────────────────
// Global raw socket fd — also used by stat reporter
static int g_sock = -1;
static struct sockaddr_in g_remoteAddr;
static bool g_remoteAddrValid = false;

void networkTask(void*) {
    // ── Create raw UDP socket ────────────────────────
    // Using raw lwIP instead of WiFiUDP for two reasons:
    // 1. We can set SO_RCVBUF to 64KB — default Arduino socket is ~8KB
    //    and overflows when a frame burst hits all at once.
    // 2. No wrapper overhead — recvfrom() is a direct lwIP syscall.
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) {
        Serial.printf("[NET] socket() failed: %d", errno);
        vTaskDelete(NULL);
        return;
    }

    // ── Increase recv buffer to 64 KB ────────────────
    // Default is ~8KB which overflows when capture.py sends a 20-chunk burst
    // in ~1ms. 64KB holds ~45 full-size chunks before dropping anything.
    int rcvbuf = 65536;
    if (setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0)
        Serial.println("[NET] SO_RCVBUF failed (non-fatal)");

    // ── Bind ─────────────────────────────────────────
    struct sockaddr_in local;
    memset(&local, 0, sizeof(local));
    local.sin_family      = AF_INET;
    local.sin_port        = htons(UDP_PORT);
    local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
        Serial.printf("[NET] bind() failed: %d", errno);
        close(g_sock);
        vTaskDelete(NULL);
        return;
    }

    // ── Non-blocking mode ────────────────────────────
    // We poll with select() so we can also do periodic beacons/stats
    // without blocking on recvfrom().
    fcntl(g_sock, F_SETFL, O_NONBLOCK);

    Serial.printf("[NET] Raw UDP socket ready on port %d, rcvbuf=%d", UDP_PORT, rcvbuf);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender;
    socklen_t senderLen = sizeof(sender);

    uint32_t lastPacketMs  = millis();
    uint32_t lastBeaconMs  = 0;
    uint32_t lastStatMs    = 0;
    uint32_t dbgPktCount   = 0;
    uint32_t dbgDropCount  = 0;   // packets received but frame already abandoned

    while (true) {
        // ── Non-blocking receive ─────────────────────
        int bytesRead = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,(struct sockaddr*)&sender, &senderLen);

        if (bytesRead < 0) {
            // No packet available (EAGAIN). Use select() with a short timeout
            // instead of portYIELD() busy-spin.
            //
            // WHY: portYIELD() in a tight loop starves IDLE0, which is what
            // resets the task watchdog → abort(). select() properly blocks the
            // task so FreeRTOS can schedule IDLE0, but wakes immediately when
            // a packet arrives — no 1ms sleep penalty like vTaskDelay(1).
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(g_sock, &rfds);
            struct timeval tv = { .tv_sec = 0, .tv_usec = 5000 }; // 5ms max block
            select(g_sock + 1, &rfds, NULL, NULL, &tv);
            // select() returns: >0 = data ready, 0 = timeout, <0 = error
            // In all cases fall through to recvfrom() at top of loop.

            // Beacon: fire max once per 2s when idle
            if ((millis() - lastBeaconMs) > 2000 && (millis() - lastPacketMs) > 2000) {
                struct sockaddr_in bcast;
                memset(&bcast, 0, sizeof(bcast));
                bcast.sin_family      = AF_INET;
                bcast.sin_port        = htons(UDP_PORT);
                bcast.sin_addr.s_addr = htonl(INADDR_BROADCAST);
                const char* beacon    = "IM HERE";
                int so = 1;
                setsockopt(g_sock, SOL_SOCKET, SO_BROADCAST, &so, sizeof(so));
                sendto(g_sock, beacon, strlen(beacon), 0,
                       (struct sockaddr*)&bcast, sizeof(bcast));
                lastBeaconMs = millis();
            }
            continue;
        }

        // ── Packet received ──────────────────────────
        lastPacketMs = millis();
        dbgPktCount++;

        if (bytesRead < 4 || rxBuf[0] != 0xAA) {
            portYIELD();
            continue;
        }

        // Record sender for replies
        memcpy(&g_remoteAddr, &sender, sizeof(sender));
        g_remoteAddrValid = true;

        // ── Control packet [0xAA 0xCC 0x01 debugState] ──
        if (rxBuf[1] == 0xCC) {
            if (bytesRead >= 4 && rxBuf[2] == 0x01) {
                debugEnabled = (rxBuf[3] == 1);
                Serial.printf("[NET] Debug %s, remote=%s:%u",debugEnabled ? "ON" : "OFF",inet_ntoa(sender.sin_addr),ntohs(sender.sin_port));}
            portYIELD();
            continue;
        }

        if (bytesRead < 8) { portYIELD(); continue; }
        int dataLen = bytesRead - 8;

        // ── Frame timeout ────────────────────────────
        if (firstChunkMs > 0 && (millis() - firstChunkMs) > FRAME_TIMEOUT_MS) {
            Serial.printf("[NET] Timeout frame=%u got=%u/%u drop=%lu",curFrameId, chunksGotCount, totalDataChunks, dbgDropCount);
            stat_framesDropped++;
            dbgDropCount = 0;
            resetFrameState();
            curFrameId = 0xFF;
        }

        // ── Data chunk [0xAA 0xBB fId cId nData nParity sHi sLo] ──
        if (rxBuf[1] == 0xBB) {
            uint8_t  fId   = rxBuf[2];
            uint8_t  cId   = rxBuf[3];
            uint8_t  nData = rxBuf[4];
            uint8_t  nPar  = rxBuf[5];
            uint16_t fSize = ((uint16_t)rxBuf[6] << 8) | rxBuf[7];

            if (fId != curFrameId) {
                if (curFrameId != 0xFF && totalDataChunks > 0) {
                    stat_framesDropped++;
                }
                resetFrameState();
                curFrameId        = fId;
                totalDataChunks   = nData;
                totalParityChunks = nPar;
                totalFrameSize    = fSize;
                firstChunkMs      = millis();
            }

            if (cId < MAX_CHUNKS && !chunkGot[cId] && dataLen > 0) {
                memcpy(chunkBuf[cId], &rxBuf[8], dataLen);
                if (dataLen < CHUNK_DATA_SIZE)
                    memset(chunkBuf[cId] + dataLen, 0, CHUNK_DATA_SIZE - dataLen);
                chunkLen[cId]  = (uint16_t)dataLen;
                chunkGot[cId]  = true;
                chunksGotCount++;
            }
        }

        // ── Parity chunk [0xAA 0xBC fId gId gStart gCount 0 0] ──
        else if (rxBuf[1] == 0xBC) {
            uint8_t fId = rxBuf[2];
            uint8_t gId = rxBuf[3];
            if (fId == curFrameId && gId < MAX_PARITY && !parityGot[gId] && dataLen > 0) {
                memcpy(parityBuf[gId], &rxBuf[8], CHUNK_DATA_SIZE);
                parityGot[gId] = true;
            } else if (fId != curFrameId) {
                dbgDropCount++;
            }
        }

        // ── Try FEC recovery if we have all data OR timeout + parity ──
        bool allData = (totalDataChunks > 0 && chunksGotCount >= totalDataChunks);
        if (!allData && totalDataChunks > 0) {
            // Try to recover missing chunks with available parity
            allData = attemptFecRecovery();
        }

        if (allData) {
            int len = assembleJpeg();
            if (len > 0) {
                uint32_t waitStart = millis();
                while (jpegReadyFlag && (millis() - waitStart) < 50) portYIELD();

                xSemaphoreTake(jpegMutex, portMAX_DELAY);
                memcpy(jpegReadyBuf, jpegAssembly, len);
                jpegReadyLen  = len;
                jpegReadyFlag = true;
                xSemaphoreGive(jpegMutex);
                xSemaphoreGive(frameSem);
                stat_framesReceived++;

                if (dbgPktCount % 100 == 0) {
                    Serial.printf("[NET] pkts=%lu frame=%u len=%d chunks=%u/%u",dbgPktCount, curFrameId, len, chunksGotCount, totalDataChunks);
                }
            }
            resetFrameState();
            curFrameId = 0xFF;
        }

        // ── Stat report via same socket (no extra WiFiUDP needed) ──
        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > 1000) {
            uint32_t elapsed = millis() - lastStatMs;
            float fps        = (float)stat_framesReceived / (elapsed / 1000.0f);
            uint32_t freeH   = ESP.getFreeHeap();
            float    ramPct  = ((float)freeH / ESP.getHeapSize()) * 100.0f;
            uint32_t freePSR = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

            snprintf(debugBuf, sizeof(debugBuf),
                "%c%cESP32-S3|FPS:%.1f|Drop:%lu|Jitter:%.1fms|DRAM:%.0f%%|PSRAM:%luKB|pkts:%lu",
                0xAB, 0xCD,
                fps, stat_framesDropped, stat_jitter, ramPct, freePSR / 1024, dbgPktCount
            );
            sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                   (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));

            stat_framesReceived = 0;
            stat_framesDropped  = 0;
            dbgPktCount         = 0;
            lastStatMs          = millis();
        }

        // Brief cooperative yield after processing a packet so other same-priority
        // tasks (including IDLE0's watchdog reset) get a chance to run.
        // portYIELD() is a single instruction, not a sleep.
        portYIELD();
    }
}
// ─────────────────────────────────────────────
//  DISPLAY STATUS HELPERS  (used during boot)
// ─────────────────────────────────────────────
static void statusLine(uint8_t row, const char* label, const char* value,
                       uint32_t col = TFT_WHITE) {
    const int LINE_H  = 22;
    const int Y_START = 58;
    int y = Y_START + row * LINE_H;
    lcd.fillRect(0, y, SCREEN_W, LINE_H, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString(label, 8, y + 3);
    lcd.setTextColor(col, TFT_BLACK);
    lcd.drawString(value, 138, y + 3);
}

static void drawBootHeader() {
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
    lcd.setTextColor(TFT_CYAN, 0x1082);
    lcd.setTextSize(2);
    lcd.drawString("ESP32-S3 STREAM", 8, 6);
    lcd.setTextSize(1);
    lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString("ILI9341  320x240  i80 parallel", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    // ── Serial: wait for USB CDC to enumerate on S3 ────
    // Without this, early prints vanish before the host port opens.
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) delay(10);
    Serial.println("\n\n[BOOT] ESP32-S3 starting...");

    // ── Display first – so every later step can show status ──
    lcd.init();
    lcd.setRotation(1);
    lcd.setColorDepth(16);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    drawBootHeader();
    statusLine(0, "Display:", "OK", TFT_GREEN);
    Serial.println("[OK] Display init");

    // ── PSRAM check ────────────────────────────────────
    bool psramOk = psramFound();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    Serial.printf("[%s] PSRAM %s\n", psramOk?"OK":"WARN", psramOk?"found":"not found");

    // ── Allocate PSRAM buffers ─────────────────────────
    chunkStorage  = (uint8_t*)heap_caps_malloc((size_t)MAX_CHUNKS * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
    parityStorage = (uint8_t*)heap_caps_malloc((size_t)MAX_PARITY * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
    jpegAssembly  = (uint8_t*)heap_caps_malloc(MAX_JPEG_SIZE,                         MALLOC_CAP_SPIRAM);
    jpegReadyBuf  = (uint8_t*)heap_caps_malloc(MAX_JPEG_SIZE,                         MALLOC_CAP_SPIRAM);

    if (!chunkStorage || !parityStorage || !jpegAssembly || !jpegReadyBuf) {
        Serial.println("[WARN] PSRAM alloc failed, falling back to DRAM");
        if (!chunkStorage)  chunkStorage  = (uint8_t*)malloc((size_t)MAX_CHUNKS * CHUNK_DATA_SIZE);
        if (!parityStorage) parityStorage = (uint8_t*)malloc((size_t)MAX_PARITY * CHUNK_DATA_SIZE);
        if (!jpegAssembly)  jpegAssembly  = (uint8_t*)malloc(MAX_JPEG_SIZE);
        if (!jpegReadyBuf)  jpegReadyBuf  = (uint8_t*)malloc(MAX_JPEG_SIZE);
        statusLine(2, "Buffers:", "DRAM fallback", TFT_YELLOW);
    } else {
        statusLine(2, "Buffers:", "PSRAM OK", TFT_GREEN);
    }

    if (!chunkStorage || !parityStorage || !jpegAssembly || !jpegReadyBuf) {
        statusLine(2, "Buffers:", "ALLOC FAILED!", TFT_RED);
        Serial.println("[ERROR] Buffer allocation failed. Halting.");
        while (1) delay(1000);
    }

    for (int i = 0; i < MAX_CHUNKS; i++) chunkBuf[i]  = chunkStorage  + (size_t)i * CHUNK_DATA_SIZE;
    for (int i = 0; i < MAX_PARITY; i++) parityBuf[i] = parityStorage + (size_t)i * CHUNK_DATA_SIZE;
    Serial.println("[OK] Buffers allocated");

    // ── Semaphores ─────────────────────────────────────
    frameSem  = xSemaphoreCreateBinary();
    jpegMutex = xSemaphoreCreateMutex();

    // ── Sprites (N16R8: force OPI PSRAM allocation) ────
    // createSprite(w, h, colorDepth, usePsram)
    // The 4th argument tells LovyanGFX to use heap_caps_malloc(MALLOC_CAP_SPIRAM)
    // instead of plain malloc(). Without this, 154KB sprites fail on DRAM.
    spr[0] = new LGFX_Sprite(&lcd);
    spr[1] = new LGFX_Sprite(&lcd);
    spr[0]->setColorDepth(16);
    spr[1]->setColorDepth(16);
    spr[0]->setPsram(true);   // force SPIRAM allocation — required for N16R8 OPI PSRAM
    spr[1]->setPsram(true);
    bool spr0ok = spr[0]->createSprite(SCREEN_W, SCREEN_H);
    bool spr1ok = spr[1]->createSprite(SCREEN_W, SCREEN_H);

    Serial.printf("[DBG] Sprite alloc: spr0=%d(%p) spr1=%d(%p)\n",
                  spr0ok, spr[0]->getBuffer(), spr1ok, spr[1]->getBuffer());

    if (!spr[0]->getBuffer() || !spr[1]->getBuffer()) {
        statusLine(3, "Sprites:", "ALLOC FAILED!", TFT_RED);
        size_t freePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        size_t freeDram  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        Serial.printf("[ERROR] Sprite alloc failed. Free PSRAM: %u  DRAM: %u\n",
                      freePsram, freeDram);
        Serial.printf("[ERROR] Need %u bytes per sprite.\n", (uint32_t)SCREEN_W * SCREEN_H * 2);
        while (1) delay(1000);
    }
    char sprBuf[40];
    snprintf(sprBuf, sizeof(sprBuf), "OK (PSRAM @%p)", spr[0]->getBuffer());
    statusLine(3, "Sprites:", sprBuf, TFT_GREEN);
    Serial.println("[OK] Sprites allocated in PSRAM");

    // ── WiFi ───────────────────────────────────────────
    // IMPORTANT for S3: call esp_wifi_set_ps AFTER WL_CONNECTED.
    // Calling it before WiFi.begin() can break the association handshake.
    statusLine(4, "WiFi:", "Connecting...", TFT_YELLOW);
    Serial.printf("[..] WiFi: connecting to \"%s\"\n", WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    uint32_t wifiStart = millis();
    uint8_t  tick = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
        Serial.print('.');

        // Animate dots on display
        char dotStr[6] = "     ";
        for (int i = 0; i < (tick % 5); i++) dotStr[i] = '.';
        char buf[32];
        snprintf(buf, sizeof(buf), "Connecting %s", dotStr);
        statusLine(4, "WiFi:", buf, TFT_YELLOW);
        tick++;

        // 20s timeout → print diagnostics, reboot
        if (millis() - wifiStart > 20000) {
            statusLine(4, "WiFi:", "TIMEOUT! Rebooting", TFT_RED);
            Serial.printf("\n[ERR] WiFi timeout. SSID=\"%s\" pass_len=%d status=%d\n",
                          WIFI_SSID, (int)strlen(WIFI_PASS), (int)WiFi.status());
            delay(3000);
            ESP.restart();
        }
    }

    // Connected – NOW disable power save
    esp_wifi_set_ps(WIFI_PS_NONE);

    String ip   = WiFi.localIP().toString();
    int    rssi = WiFi.RSSI();
    char   ipBuf[36];
    snprintf(ipBuf, sizeof(ipBuf), "%s (%d dBm)", ip.c_str(), rssi);
    statusLine(4, "WiFi:", ipBuf, TFT_GREEN);
    statusLine(5, "UDP port:", String(UDP_PORT).c_str(), TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
    Serial.printf("\n[OK] WiFi: %s  RSSI: %d dBm\n", ip.c_str(), rssi);

    // ── Network task on Core 0 ─────────────────────────
    xTaskCreatePinnedToCore(networkTask, "NetTask", 8192, NULL, 2, NULL, 0);

    stat_lastReport = millis();
    Serial.println("[OK] Setup complete. Waiting for stream...");
}
// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 – rendering)
// ─────────────────────────────────────────────
static uint8_t sprIdx = 0;

// ─────────────────────────────────────────────
//  JPEGDEC  (SIMD-accelerated on ESP32-S3)
// ─────────────────────────────────────────────
// One JPEGDEC instance per sprite (decode into back-buffer while front pushes)
static JPEGDEC jpeg[2];

// Callback: JPEGDEC calls this for each 16×16 MCU block as it decodes.
// We write directly into the sprite's pixel buffer — zero copy.
// This also means the i80 DMA push of sprite[front] runs on the bus
// while Core 1 is here decoding sprite[back] — true overlap.
static int jpegDrawCallback(JPEGDRAW* pDraw) {
    // pDraw->pUser points to the target LGFX_Sprite
    LGFX_Sprite* dst = (LGFX_Sprite*)pDraw->pUser;
    // pushImage handles the blit of one MCU block into the sprite framebuffer
    dst->pushImage(pDraw->x, pDraw->y, pDraw->iWidth, pDraw->iHeight,
                   (uint16_t*)pDraw->pPixels);
    return 1; // 1 = continue decoding, 0 = abort
}

// Decode a JPEG from PSRAM into sprite[idx].
// Returns decode time in microseconds, or 0 on failure.
static uint32_t decodeJpegToSprite(uint8_t* jpegData, int jpegLen, uint8_t idx) {
    // JPEGDEC requires the source buffer to be 4-byte aligned.
    // Our PSRAM slab is malloc'd so this is guaranteed, but assert it anyway.
    if ((uintptr_t)jpegData & 3) {
        Serial.println("[JPEG] Buffer not 4-byte aligned!");
        return 0;
    }

    uint32_t t0 = micros();

    // openRAM: decode from a RAM buffer (not flash/SD).
    // jpegDrawCallback writes blocks into the sprite.
    // pUser = pointer to the sprite we're targeting.
    if (!jpeg[idx].openRAM(jpegData, jpegLen, jpegDrawCallback)) {
        Serial.printf("[JPEG] openRAM failed, rc=%d\n", jpeg[idx].getLastError());
        return 0;
    }

    // RGB565_BIG_ENDIAN: LovyanGFX i80 parallel bus wants big-endian 16-bit pixels.
    // SIMD color conversion is active automatically when ESP32S3 is defined.
    jpeg[idx].setPixelType(RGB565_BIG_ENDIAN);
    jpeg[idx].setUserPointer((void*)spr[idx]);  // FIX: pDraw->pUser in callback was NULL without this

    spr[idx]->fillScreen(TFT_BLACK);
    int rc = jpeg[idx].decode(0, 0, 0);  // x=0, y=0, options=0 (no scaling/flip)
    jpeg[idx].close();

    if (rc == 0) {
        Serial.printf("[JPEG] decode failed, rc=%d\n", jpeg[idx].getLastError());
        return 0;
    }

    return micros() - t0;
}

void loop() {
    // Wait up to one frame period for a new JPEG
    if (xSemaphoreTake(frameSem, pdMS_TO_TICKS(40)) == pdTRUE) {

        // Jitter tracking
        uint32_t now = millis();
        if (stat_prevFrameMs > 0) {
            static uint32_t lastInterval = 0;
            uint32_t iv = now - stat_prevFrameMs;
            if (lastInterval > 0) {
                int32_t diff = (int32_t)iv - (int32_t)lastInterval;
                stat_jitter += (fabsf((float)diff) - stat_jitter) / 16.0f;
            }
            lastInterval = iv;
        }
        stat_prevFrameMs = now;

        // Snapshot len+flag under mutex, then release before slow JPEG decode
        xSemaphoreTake(jpegMutex, portMAX_DELAY);
        int  len   = jpegReadyLen;
        bool valid = jpegReadyFlag;
        jpegReadyFlag = false;     // mark consumed so network task can write next frame
        xSemaphoreGive(jpegMutex);

        if (valid && len > 0) {
            static bool streamStarted = false;
            static uint32_t decodeTotal = 0;
            static uint32_t decodeCount = 0;

            // ── OVERLAP STRATEGY ─────────────────────────────
            // back  = sprIdx        → we decode new frame into this
            // front = sprIdx ^ 1    → currently pushed to display
            //
            // Timeline:
            //   [Core 1] decode frame N into sprite[back]  ← happening now
            //   [i80 bus] pushSprite(front) is DMA/parallel ← runs simultaneously
            //
            // Because LovyanGFX i80 pushSprite is non-blocking (bus runs
            // independently), the CPU is free to decode into the other sprite.
            // This is why block-callback JPEGDEC beats whole-frame decoders here.

            // First, kick off the push of the PREVIOUS frame (now front) to screen.
            // This starts the i80 bus transfer and returns immediately.
            if (streamStarted) {
                spr[sprIdx ^ 1]->pushSprite(0, 0);
            }

            // Now decode new frame into the back sprite while bus is busy.
            uint32_t dt = decodeJpegToSprite(jpegReadyBuf, len, sprIdx);

            if (dt == 0) {
                Serial.printf("[RENDER] decode FAILED len=%d\n", len);
            } else {
                decodeTotal += dt;
                decodeCount++;

                if (!streamStarted) {
                    streamStarted = true;
                    // First frame: we have nothing in front yet, push back immediately
                    spr[sprIdx]->pushSprite(0, 0);
                    statusLine(6, "Status:", "STREAMING!", TFT_GREEN);
                    Serial.printf("[RENDER] First frame OK: %d bytes, decode=%lu us\n",
                                  len, dt);
                    delay(300);
                }

                // Log average decode time every 60 frames
                if (decodeCount % 60 == 0) {
                    Serial.printf("[RENDER] avg decode: %lu us (%.1f ms)\n",
                                  decodeTotal / decodeCount,
                                  (decodeTotal / decodeCount) / 1000.0f);
                    decodeTotal = 0; decodeCount = 0;
                }

                // Swap front/back
                sprIdx ^= 1;
            }
        }
    }
    // Stat reporting is in networkTask (uses same udp socket)
}