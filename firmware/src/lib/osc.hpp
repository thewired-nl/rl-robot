#pragma once

#include <WiFi.h>
#include <WiFiUdp.h>
#include <vector>
#include <functional>

#include "OSCMessage.h"

#include "log.h"

// typedef void (*RouteFunction)(OSCMessage &, IPAddress);
typedef std::function<void(OSCMessage&, IPAddress)> RouteFunction;

struct OSCRoute {
	char const * route;
	RouteFunction func;
};

namespace OSC {
	Log log("osc", ANSI_GREEN);
	WiFiUDP udp;
	WiFiUDP udp_send;
	std::vector<OSCRoute> routes;
	IPAddress remote{192, 168, 1, 142};

	void listen() {
		int err;
		if ((err = udp.begin(1212)) > 0) {
			log("Listening on %s:1212", WiFi.localIP().toString().c_str());
		} else {
			log.error("Failed to open port 1212: %d", err);
		}
	}

	void addRoute(const char *route, RouteFunction func) {
		log("added handler for route %s", route);
		routes.push_back(OSCRoute{route, func});
	}

	bool sendMessage(OSCMessage &msg, IPAddress ip = remote) {
		udp_send.beginPacket(ip, 1234);
		msg.send(udp_send);
		return udp_send.endPacket();
	}

	void loop() {
		int packetSize = udp.parsePacket();
		if (packetSize) {
			// log("Received packet from %s (len=%d)", udp.remoteIP().toString().c_str(), packetSize);

			OSCMessage msg;
			while (packetSize--) {
				msg.fill(udp.read());
			}

			if (msg.hasError()) {
				log.error("Error decoding (%d)", msg.getError());
			} else {
				bool matched = false;
				for (OSCRoute r : routes) {
					if (msg.fullMatch(r.route)) {
						remote = udp.remoteIP();
						r.func(msg, udp.remoteIP());
						matched = true;
						break;
					}
					// if (msg.dispatch(r.route, r.func)) {
					// 	matched = true;
					// 	break;
					// }
				}
				if (!matched) {
					char addr[64];
					msg.getAddress(addr);
					log("'%s' matched no routes", addr);
				}
			}
		}
	}
}
