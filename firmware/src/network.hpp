#pragma once

#include <IPAddress.h>
#include <OSCMessage.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>

#include "lib/log.h"
#include "lib/task_cpp.hpp"
#include "secrets.h"
#include "lib/osc.hpp"

WiFiUDP udp;
WiFiMulti wifiMulti;

class NetworkTask : public TaskClass {
  Log log{"network", ANSI_PURPLE};

 public:
  NetworkTask() : TaskClass("network", 4, 20000) {}

  bool connected = false;
  bool isReconnect = true;
  bool firstConnect = true;

	void setup() {
		WiFi.persistent(true);
	
		log("configured WiFi networks:");
		for (auto w : wifiNetworks) {
			if (strlen(w[0]) > 0) {
				log("  - %s", w[0]);
				wifiMulti.addAP(w[0], w[1]);
			}
		}
	}

  void task() {
		for (;;) {
			if (wifiMulti.run() != WL_CONNECTED) {
				connected = false;
				isReconnect = true;
				log.error("lost WiFi... reconnecting");
				delay(1000);
				continue;
			} else if (isReconnect) {
				connected = true;
				isReconnect = false;
				log("connected to %s as %s", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
	
				OSC::listen();
			}
	
			OSC::loop();
			vTaskDelay(1);
		}
  }
};
