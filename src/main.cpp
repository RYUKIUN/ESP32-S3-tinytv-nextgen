/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * TILE ARCHITECTURE:
 *   Frame split into 4 independent 160x120 quadrants on the PC.
 *   Each quadrant encoded as its own JPEG and sent independently.
 *   ESP32 decodes each tile as it arrives and blits to the correct region.
 *   A lost/corrupt tile leaves old pixels in place -- no whole-frame corruption.
 *
 *   Tile layout:
 *     [0: TL] [1: TR]
 *     [2: BL] [3: BR]
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
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>

// ─────────────────────────────────────────────
//  DISPLAY CONFIGURATION
// ─────────────────────────────────────────────
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_Parallel8  _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        {
            auto cfg       = _bus.config();
            cfg.freq_write = 20000000;
            cfg.pin_wr = 1;  cfg.pin_rd = 40; cfg.pin_rs = 2;
            cfg.pin_d0 = 5;  cfg.pin_d1 =  4; cfg.pin_d2 = 10;
            cfg.pin_d3 = 9;  cfg.pin_d4 =  3; cfg.pin_d5 =  8;
            cfg.pin_d6 = 7;  cfg.pin_d7 =  6;
            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
        {
            auto cfg = _panel.config();
            cfg.pin_cs  = 41; cfg.pin_rst = 39; cfg.pin_busy = -1;
            cfg.panel_width = 240; cfg.panel_height = 320;
            cfg.offset_x = 0; cfg.offset_y = 0; cfg.offset_rotation = 0;
            cfg.dummy_read_pixel = 8;
            cfg.readable = false; cfg.invert = false;
            cfg.rgb_order = false; cfg.dlen_16bit = false; cfg.bus_shared = false;
            _panel.config(cfg);
        }
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

#define SCREEN_W        320
#define SCREEN_H        240
#define NUM_TILES       4
#define TILE_W          160
#define TILE_H          120
#define CHUNK_DATA_SIZE 1400
#define MAX_TILE_CHUNKS 24     // 24x1400 = 33.6 KB max per tile JPEG
#define MAX_TILE_JPEG   (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)
#define TILE_TIMEOUT_MS 200

static const int16_t TILE_X[NUM_TILES] = {   0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {   0,   0, 120, 120 };

// ─────────────────────────────────────────────
//  TILE REASSEMBLY STATE
// ─────────────────────────────────────────────
struct TileState {
    uint8_t* chunkBuf[MAX_TILE_CHUNKS];
    uint16_t chunkLen[MAX_TILE_CHUNKS];
    bool     chunkGot[MAX_TILE_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint16_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    uint8_t* assembly     = nullptr;
    int      readyLen     = 0;
    bool     readyFlag    = false;
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};

static TileState tiles[NUM_TILES];
static uint8_t*  tileChunkStorage[NUM_TILES] = {};
SemaphoreHandle_t tileMutex[NUM_TILES];
SemaphoreHandle_t tileSem[NUM_TILES];

// ─────────────────────────────────────────────
//  STATS
// ─────────────────────────────────────────────
static bool     debugEnabled     = false;
static char     debugBuf[400];
static int      g_sock           = -1;
static struct sockaddr_in g_remoteAddr;
static bool     g_remoteAddrValid = false;
static float    stat_jitter      = 0.0f;
static uint32_t stat_prevMs      = 0;

// ─────────────────────────────────────────────
//  TILE HELPERS
// ─────────────────────────────────────────────
static void resetTile(uint8_t t) {
    memset(tiles[t].chunkGot, 0, sizeof(tiles[t].chunkGot));
    tiles[t].frameId      = 0xFF;
    tiles[t].totalChunks  = 0;
    tiles[t].frameSize    = 0;
    tiles[t].chunksGot    = 0;
    tiles[t].firstChunkMs = 0;
}

static int assembleTile(uint8_t t) {
    TileState& ts = tiles[t];
    if (!ts.assembly || ts.totalChunks == 0) return 0;
    int offset = 0;
    for (uint8_t c = 0; c < ts.totalChunks; c++) {
        if (!ts.chunkGot[c]) return 0;
        memcpy(ts.assembly + offset, ts.chunkBuf[c], ts.chunkLen[c]);
        offset += ts.chunkLen[c];
    }
    if (offset < 4 ||
        ts.assembly[0] != 0xFF || ts.assembly[1] != 0xD8 ||
        ts.assembly[offset-2] != 0xFF || ts.assembly[offset-1] != 0xD9) {
        Serial.printf("[TILE%u] Bad markers len=%d [%02X%02X..%02X%02X]\n", t, offset,
            ts.assembly[0], ts.assembly[1], ts.assembly[offset-2], ts.assembly[offset-1]);
        ts.stat_corrupt++;
        return 0;
    }
    return offset;
}

// ─────────────────────────────────────────────
//  JPEGDEC
// ─────────────────────────────────────────────
static JPEGDEC jpeg_dec;

struct TileDrawCtx { int16_t xOff, yOff; };
static TileDrawCtx tileDrawCtx;

static int tileJpegCallback(JPEGDRAW* pDraw) {
    lcd.pushImage(
        tileDrawCtx.xOff + pDraw->x,
        tileDrawCtx.yOff + pDraw->y,
        pDraw->iWidth, pDraw->iHeight,
        (uint16_t*)pDraw->pPixels
    );
    return 1;
}

static bool decodeTileToLcd(uint8_t t, uint8_t* buf, int len) {
    if ((uintptr_t)buf & 3) { Serial.printf("[TILE%u] unaligned\n",t); return false; }
    tileDrawCtx = { TILE_X[t], TILE_Y[t] };
    if (!jpeg_dec.openRAM(buf, len, tileJpegCallback)) {
        Serial.printf("[TILE%u] open err=%d\n", t, jpeg_dec.getLastError());
        return false;
    }
    jpeg_dec.setPixelType(RGB565_BIG_ENDIAN);
    jpeg_dec.setUserPointer(&tileDrawCtx);
    int rc = jpeg_dec.decode(0, 0, 0);
    jpeg_dec.close();
    if (!rc) { Serial.printf("[TILE%u] decode err=%d len=%d\n",t,jpeg_dec.getLastError(),len); return false; }
    return true;
}

// ─────────────────────────────────────────────
//  NETWORK TASK  (Core 0)
// ─────────────────────────────────────────────
// Packet: [0xAA, 0xBB, frame_id, tile_id, chunk_id, total_chunks, size_hi, size_lo] + data
// Control:[0xAA, 0xCC, 0x01, debug_state]
void networkTask(void*) {
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) { Serial.println("[NET] socket fail"); vTaskDelete(NULL); return; }
    int rcvbuf = 65536;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    struct sockaddr_in local = {};
    local.sin_family = AF_INET; local.sin_port = htons(UDP_PORT); local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock,(struct sockaddr*)&local,sizeof(local))<0) {
        Serial.println("[NET] bind fail"); close(g_sock); vTaskDelete(NULL); return;
    }
    fcntl(g_sock, F_SETFL, O_NONBLOCK);
    Serial.printf("[NET] UDP ready port=%d\n", UDP_PORT);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender; socklen_t slen = sizeof(sender);
    uint32_t lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0, pktCount = 0;

    while (true) {
        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0, (struct sockaddr*)&sender, &slen);

        if (n < 0) {
            fd_set rfds; FD_ZERO(&rfds); FD_SET(g_sock, &rfds);
            struct timeval tv = { .tv_sec = 0, .tv_usec = 5000 };
            select(g_sock+1, &rfds, NULL, NULL, &tv);
            if ((millis()-lastBeaconMs)>2000 && (millis()-lastPktMs)>2000) {
                struct sockaddr_in bc = {}; bc.sin_family = AF_INET;
                bc.sin_port = htons(UDP_PORT); bc.sin_addr.s_addr = htonl(INADDR_BROADCAST);
                int so=1; setsockopt(g_sock,SOL_SOCKET,SO_BROADCAST,&so,sizeof(so));
                const char* b="ESP32_READY"; sendto(g_sock,b,strlen(b),0,(struct sockaddr*)&bc,sizeof(bc));
                lastBeaconMs = millis();
            }
            continue;
        }

        lastPktMs = millis(); pktCount++;
        if (n < 4 || rxBuf[0] != 0xAA) { portYIELD(); continue; }
        memcpy(&g_remoteAddr, &sender, sizeof(sender)); g_remoteAddrValid = true;

        // Control
        if (rxBuf[1] == 0xCC) {
            if (n>=4 && rxBuf[2]==0x01) { debugEnabled=(rxBuf[3]==1); }
            portYIELD(); continue;
        }

        // Tile chunk
        if (rxBuf[1] != 0xBB || n < 8) { portYIELD(); continue; }
        uint8_t  fId     = rxBuf[2];
        uint8_t  tId     = rxBuf[3];
        uint8_t  cId     = rxBuf[4];
        uint8_t  nChunks = rxBuf[5];
        uint16_t fSize   = ((uint16_t)rxBuf[6]<<8)|rxBuf[7];
        int      dataLen = n - 8;
        if (tId >= NUM_TILES || dataLen <= 0) { portYIELD(); continue; }

        TileState& ts = tiles[tId];

        // Timeout stale tile
        if (ts.firstChunkMs>0 && (millis()-ts.firstChunkMs)>TILE_TIMEOUT_MS) {
            Serial.printf("[TILE%u] timeout got=%u/%u\n", tId, ts.chunksGot, ts.totalChunks);
            ts.stat_timeout++; resetTile(tId);
        }

        // New tile frame
        if (fId != ts.frameId) { resetTile(tId); ts.frameId=fId; ts.totalChunks=nChunks; ts.frameSize=fSize; ts.firstChunkMs=millis(); }

        // Store chunk
        if (cId < MAX_TILE_CHUNKS && !ts.chunkGot[cId]) {
            memcpy(ts.chunkBuf[cId], &rxBuf[8], dataLen);
            ts.chunkLen[cId]=(uint16_t)dataLen; ts.chunkGot[cId]=true; ts.chunksGot++;
        }

        // Complete?
        if (ts.chunksGot >= ts.totalChunks) {
            int len = assembleTile(tId);
            if (len > 0) {
                uint32_t ws=millis();
                while (ts.readyFlag && (millis()-ws)<30) portYIELD();
                xSemaphoreTake(tileMutex[tId], portMAX_DELAY);
                ts.readyLen=len; ts.readyFlag=true;
                xSemaphoreGive(tileMutex[tId]);
                xSemaphoreGive(tileSem[tId]);
            }
            resetTile(tId);
        }

        // Stats
        if (debugEnabled && g_remoteAddrValid && (millis()-lastStatMs)>1000) {
            uint32_t el = millis()-lastStatMs;
            uint32_t totalDec = 0;
            for (int i=0;i<NUM_TILES;i++) totalDec += tiles[i].stat_decoded;
            float fps = (float)totalDec/(el/1000.0f);
            uint32_t freeH=ESP.getFreeHeap(); float ramPct=(float)freeH/ESP.getHeapSize()*100.0f;
            uint32_t freePSR=heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            snprintf(debugBuf,sizeof(debugBuf),
                "%c%cFPS:%.1f|JIT:%.1fms|DRAM:%.0f%%|PSRAM:%luKB|PKTS:%lu"
                "|T0:ok=%lu,c=%lu,to=%lu|T1:ok=%lu,c=%lu,to=%lu"
                "|T2:ok=%lu,c=%lu,to=%lu|T3:ok=%lu,c=%lu,to=%lu",
                0xAB,0xCD,fps,stat_jitter,ramPct,freePSR/1024,pktCount,
                tiles[0].stat_decoded,tiles[0].stat_corrupt,tiles[0].stat_timeout,
                tiles[1].stat_decoded,tiles[1].stat_corrupt,tiles[1].stat_timeout,
                tiles[2].stat_decoded,tiles[2].stat_corrupt,tiles[2].stat_timeout,
                tiles[3].stat_decoded,tiles[3].stat_corrupt,tiles[3].stat_timeout);
            sendto(g_sock,debugBuf,strlen(debugBuf),0,(struct sockaddr*)&g_remoteAddr,sizeof(g_remoteAddr));
            for (int i=0;i<NUM_TILES;i++) tiles[i].stat_decoded=tiles[i].stat_corrupt=tiles[i].stat_timeout=0;
            pktCount=0; lastStatMs=millis();
        }
        portYIELD();
    }
}

// ─────────────────────────────────────────────
//  DISPLAY STATUS HELPERS
// ─────────────────────────────────────────────
static void statusLine(uint8_t row, const char* label, const char* value, uint32_t col=TFT_WHITE) {
    int y = 58+row*22;
    lcd.fillRect(0,y,SCREEN_W,22,TFT_BLACK);
    lcd.setTextColor(0x7BEF,TFT_BLACK); lcd.drawString(label,8,y+3);
    lcd.setTextColor(col,TFT_BLACK);    lcd.drawString(value,138,y+3);
}
static void drawBootHeader() {
    lcd.fillScreen(TFT_BLACK); lcd.setTextFont(2); lcd.setTextSize(1);
    lcd.fillRect(0,0,SCREEN_W,54,0x1082);
    lcd.setTextColor(TFT_CYAN,0x1082); lcd.setTextSize(2); lcd.drawString("ESP32-S3 STREAM",8,6);
    lcd.setTextSize(1); lcd.setTextColor(0x7BEF,0x1082);
    lcd.drawString("ILI9341  320x240  4-tile mode",8,34);
    lcd.drawFastHLine(0,54,SCREEN_W,TFT_DARKGREY);
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    uint32_t t0=millis(); while(!Serial&&(millis()-t0)<2000) delay(10);
    Serial.println("\n[BOOT] 4-tile mode");

    lcd.init(); lcd.setRotation(1); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    drawBootHeader(); statusLine(0,"Display:","OK",TFT_GREEN);

    bool psramOk=psramFound();
    statusLine(1,"PSRAM:",psramOk?"Found":"MISSING!",psramOk?TFT_GREEN:TFT_RED);

    // Allocate per-tile PSRAM buffers
    bool allocOk=true;
    for (int t=0;t<NUM_TILES;t++) {
        tileChunkStorage[t]=(uint8_t*)heap_caps_malloc((size_t)MAX_TILE_CHUNKS*CHUNK_DATA_SIZE,MALLOC_CAP_SPIRAM);
        if (!tileChunkStorage[t]) tileChunkStorage[t]=(uint8_t*)malloc((size_t)MAX_TILE_CHUNKS*CHUNK_DATA_SIZE);
        tiles[t].assembly=(uint8_t*)heap_caps_malloc(MAX_TILE_JPEG,MALLOC_CAP_SPIRAM);
        if (!tiles[t].assembly) tiles[t].assembly=(uint8_t*)malloc(MAX_TILE_JPEG);
        if (!tileChunkStorage[t]||!tiles[t].assembly) { allocOk=false; break; }
        for (int c=0;c<MAX_TILE_CHUNKS;c++) tiles[t].chunkBuf[c]=tileChunkStorage[t]+(size_t)c*CHUNK_DATA_SIZE;
        tileMutex[t]=xSemaphoreCreateMutex();
        tileSem[t]=xSemaphoreCreateBinary();
    }
    if (!allocOk) { statusLine(2,"Buffers:","ALLOC FAILED!",TFT_RED); while(1) delay(1000); }
    statusLine(2,"Buffers:","4x tile PSRAM OK",TFT_GREEN);

    // WiFi
    statusLine(3,"WiFi:","Connecting...",TFT_YELLOW);
    WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.begin(WIFI_SSID,WIFI_PASS);
    uint32_t ws=millis(); uint8_t tick=0;
    while (WiFi.status()!=WL_CONNECTED) {
        delay(250); tick++;
        char buf[24]; snprintf(buf,sizeof(buf),"Conn%.*s",tick%5,".....");
        statusLine(3,"WiFi:",buf,TFT_YELLOW);
        if (millis()-ws>20000) { statusLine(3,"WiFi:","TIMEOUT!",TFT_RED); delay(3000); ESP.restart(); }
    }
    esp_wifi_set_ps(WIFI_PS_NONE);
    String ip=WiFi.localIP().toString(); char ipBuf[36];
    snprintf(ipBuf,sizeof(ipBuf),"%s (%ddBm)",ip.c_str(),WiFi.RSSI());
    statusLine(3,"WiFi:",ipBuf,TFT_GREEN);
    statusLine(4,"UDP:",String(UDP_PORT).c_str(),TFT_CYAN);
    statusLine(5,"Mode:","4-tile independent",TFT_CYAN);
    statusLine(6,"Status:","Waiting for PC...",TFT_YELLOW);
    Serial.printf("[OK] WiFi: %s\n",ip.c_str());

    xTaskCreatePinnedToCore(networkTask,"NetTask",8192,NULL,2,NULL,0);
    Serial.println("[OK] Ready.");
}

// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1)
// ─────────────────────────────────────────────
void loop() {
    static bool     streamStarted = false;
    static uint32_t decodeTotal   = 0;
    static uint32_t decodeCount   = 0;
    bool anyRendered = false;

    for (uint8_t t=0; t<NUM_TILES; t++) {
        if (xSemaphoreTake(tileSem[t],0) != pdTRUE) continue;

        xSemaphoreTake(tileMutex[t],portMAX_DELAY);
        int  len  = tiles[t].readyLen;
        bool flag = tiles[t].readyFlag;
        tiles[t].readyFlag = false;
        xSemaphoreGive(tileMutex[t]);

        if (!flag || len<=0) continue;

        uint32_t tDec = micros();
        bool ok = decodeTileToLcd(t, tiles[t].assembly, len);
        tDec = micros()-tDec;

        if (ok) {
            tiles[t].stat_decoded++;
            decodeTotal+=tDec; decodeCount++;
            anyRendered = true;
            if (!streamStarted) {
                streamStarted=true;
                statusLine(6,"Status:","STREAMING!",TFT_GREEN);
                Serial.printf("[RENDER] First tile %u: %d bytes %lu us\n",t,len,tDec);
                delay(200);
            }
        }
        // Failed: old pixels stay -- intentional

        // Jitter
        uint32_t now=millis();
        if (stat_prevMs>0) {
            static uint32_t lastIv=0; uint32_t iv=now-stat_prevMs;
            if (lastIv>0) { int32_t d=(int32_t)iv-(int32_t)lastIv; stat_jitter+=(fabsf((float)d)-stat_jitter)/16.0f; }
            lastIv=iv;
        }
        stat_prevMs=now;

        if (decodeCount>0 && decodeCount%120==0) {
            Serial.printf("[RENDER] avg tile: %lu us\n", decodeTotal/decodeCount);
            decodeTotal=0; decodeCount=0;
        }
    }
    if (!anyRendered) vTaskDelay(1);
}