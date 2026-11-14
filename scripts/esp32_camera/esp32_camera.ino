// AI Thinker ESP32-CAM pinout. Confirm the physical board before uploading.
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
httpd_handle_t server = nullptr;

esp_err_t stream(httpd_req_t* request) {
  esp_err_t result = httpd_resp_set_type(request, "multipart/x-mixed-replace; boundary=frame");
  while (result == ESP_OK) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (!frame) return ESP_FAIL;
    char header[96];
    int length = snprintf(header, sizeof(header),
        "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
        static_cast<unsigned>(frame->len));
    result = httpd_resp_send_chunk(request, header, length);
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, reinterpret_cast<const char*>(frame->buf), frame->len);
    }
    esp_camera_fb_return(frame);
    delay(30);
  }
  return result;
}

void setup() {
  Serial.begin(115200);
  camera_config_t c = {};
  c.pin_pwdn = 32; c.pin_reset = -1; c.pin_xclk = 0;
  c.pin_sccb_sda = 26; c.pin_sccb_scl = 27;
  c.pin_d0 = 5; c.pin_d1 = 18; c.pin_d2 = 19; c.pin_d3 = 21;
  c.pin_d4 = 36; c.pin_d5 = 39; c.pin_d6 = 34; c.pin_d7 = 35;
  c.pin_vsync = 25; c.pin_href = 23; c.pin_pclk = 22;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;
  c.jpeg_quality = 12;
  bool psram = psramFound();
  c.fb_count = psram ? 2 : 1;
  c.fb_location = psram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = psram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t camera = esp_camera_init(&c);
  if (camera != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", camera);
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) delay(250);
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi connection failed. Check SSID/password and reset.");
    return;
  }
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  httpd_uri_t route = {};
  route.uri = "/stream";
  route.method = HTTP_GET;
  route.handler = stream;
  if (httpd_start(&server, &config) != ESP_OK ||
      httpd_register_uri_handler(server, &route) != ESP_OK) {
    Serial.println("Cannot start stream server.");
    return;
  }
  Serial.printf("Stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() { delay(1000); }
