#include "log.h"
#include <cstdarg>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/_stdint.h>
#include <vector>
#include "HardwareSerial.h"

float time() {
	return millis()/10000.0;
}

Log::Log(const char* name, const char* color) {
	snprintf(_name, sizeof(_name), "%s[%s]%s", color, name, ANSI_DEFAULT);
	snprintf(_errorName, sizeof(_errorName), "%s[%s:%sERROR%s]%s", color, name, ANSI_RED, color, ANSI_DEFAULT);
}

static const char* const final_format = "%s~%7.3f%s %s %s\r\n"; // ~0.00 [name] str\r\n

void Log::write(const char* fmt, va_list arg, bool isError) {
  Serial.printf("%s~%7.3f%s%s ", ANSI_REDUCED, time(), ANSI_DEFAULT, isError ? _errorName : _name);
  Serial.vprintf(fmt, arg);
  Serial.println();
}

// void Log::write(const char* fmt, va_list arg, bool isError) {
// 	va_list copy;
// 	va_copy(copy, arg);
// 	// initialize vector with size (1 + length)
// 	std::vector<char> buf(1 + vsnprintf(nullptr, 0, fmt, arg));
// 	vsnprintf(buf.data(), buf.size(), fmt, copy);
// 	va_end(copy);

// 	std::vector<char> buf2(1 + snprintf(nullptr, 0, final_format,
// 		ANSI_REDUCED,
// 		time(),
// 		ANSI_DEFAULT,
// 		isError
// 			? _errorName
// 			: _name,
// 		buf.data()
// 	));

// 	snprintf(buf2.data(), buf2.size(), final_format,
// 		ANSI_REDUCED,
// 		time(),
// 		ANSI_DEFAULT,
// 		isError
// 			? _errorName
// 			: _name,
// 		buf.data()
// 	);

// 	if (Serial0.availableForWrite()) {
// 		Serial0.write(buf2.data(), buf2.size());
// 	}
// }

void Log::operator()(const char* fmt, ...) {
	va_list arg;
	va_start(arg, fmt);
	Log::write(fmt, arg);
	va_end(arg);
}

void Log::error(const char* fmt, ...) {
	va_list arg;
	va_start(arg, fmt);
	Log::write(fmt, arg, true);
	va_end(arg);
}
