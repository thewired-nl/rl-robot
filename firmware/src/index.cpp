// #define MODEL_PID

#include <Wire.h>
#include <esp_timer.h>

#include "inputs.hpp"
#include "lib/osc.hpp"
#include "log.h"
#include "model.hpp"
#include "network.hpp"
#include "pid.hpp"

// constexpr float TIMESTEP = 0.008f; // 8ms
constexpr float TIMESTEP = 0.005f; // 5ms

Log logger("balancer", ANSI_PURPLE);

#ifdef MODEL_PID
PIDController model;
#else
MLModel model;
#endif

InputCollector ic;
NetworkTask network;

// IPAddress _ip{10, 42, 45, 111};

Log log_local("localinfer", ANSI_BLUE);

void setup() {
  Serial0.begin(115200);
  delay(100);
  logger("boot");
  // Serial.setDebugOutput(true);

  Wire.setPins(13, 14);

  model.setup();

  ic.setup();

  network.setup();

  while (!network.connected) {
    delay(500);
    if (!network.connected) {
      logger("waiting for network..");
    }
  }

  // delay(500);

  // ic.actTask.drive(1.0, 1.0, 1.0);

  // OSCMessage msg("/motor");

  // for (int i = 0; i < 500; i++) {
  //   unsigned long loop_start = micros();
  //   ic.update(model.input);
  //   msg.empty();
  //   msg.add(Encoder::normalize(model.input[3]));
  //   msg.add(Encoder::normalize(model.input[4]));
  //   msg.add((float)model.input[7]);
  //   msg.add((float)model.input[8]);
  //   OSC::sendMessage(msg, _ip);

  //   while (micros() - loop_start < (unsigned long)(TIMESTEP * 1e6));
  // }

  // delay(1000);

  // ic.actTask.drive(0.0, 0.0, 1.0);

  // for (int i = 0; i < 500; i++) {
  //   unsigned long loop_start = micros();
  //   ic.update(model.input);
  //   msg.empty();
  //   msg.add(Encoder::normalize(model.input[3]));
  //   msg.add(Encoder::normalize(model.input[4]));
  //   msg.add((float)model.input[7]);
  //   msg.add((float)model.input[8]);
  //   OSC::sendMessage(msg, _ip);

  //   while (micros() - loop_start < (unsigned long)(TIMESTEP * 1e6));
  // }

  // delay(1000);

  // for (float j = 0; j < 1.0; j += 0.2) {
  //   ic.actTask.drive(j, j, 1.0);

  //   for (int i = 0; i < 200; i++) {
  //     unsigned long loop_start = micros();
  //     ic.update(model.input);
  //     msg.empty();
  //     msg.add(Encoder::normalize(model.input[3]));
  //     msg.add(Encoder::normalize(model.input[4]));
  //     msg.add((float)model.input[7]);
  //     msg.add((float)model.input[8]);
  //     OSC::sendMessage(msg, _ip);

  //     while (micros() - loop_start < (unsigned long)(TIMESTEP * 1e6));
  //   }
  // }

  // delay(1000);

  // bool j = false;
  // for (int i = 0; i < 500; i++) {
  //   if (i % 50 == 0) {
  //     if (j) {
  //       ic.actTask.drive(0.8, 0.8, 1.0);
  //     } else {
  //       ic.actTask.drive(-0.8, -0.8, 1.0);
  //     }
  //     j = !j;
  //   }
  //   unsigned long loop_start = micros();
  //   ic.update(model.input);
  //   msg.empty();
  //   msg.add(Encoder::normalize(model.input[3]));
  //   msg.add(Encoder::normalize(model.input[4]));
  //   msg.add((float)model.input[7]);
  //   msg.add((float)model.input[8]);
  //   OSC::sendMessage(msg, _ip);

  //   while (micros() - loop_start < (unsigned long)(TIMESTEP * 1e6));
  // }

  // delay(1000);
  // ic.actTask.drive(0.0, 0.0, 1.0);

  // delay(10000000);
}

OSCMessage osc_obs("/obs");
void loop() {
  unsigned long loop_start = micros();

  // [pitch, pitch_rate, yaw_rate, cmd_vel, cmd_yaw]
  // model.input[0] = 0.0;
  // model.input[1] = 0.0;
  // model.input[2] = 0.0;
  // model.input[0] = eul.z();
  // model.input[1] = gyro.x();
  // model.input[2] = gyro.z();
  // model.input[3] = 0.0;
  // model.input[4] = 0.0;

  ic.update(model.input);
  osc_obs.empty();
  for (int i = 0; i < 9; i++) {
    osc_obs.add(model.input[i]);
  }
  OSC::sendMessage(osc_obs);

  if (ic.cmdTask.switchA) {
    model.run();
    // log_local("[ %4.2f %4.2f ]", model.output[0], model.output[1]);
  } else {
    // manual control
    model.output[0] = ic.cmdTask.input[0];
    model.output[1] = ic.cmdTask.input[0];
  }

  if (abs(model.input[0]) >= 0.53) {
    logger.error("ive fallen and i cant get up :(");
    model.output[0] = 0.0;
    model.output[1] = 0.0;
  }

  ic.actTask.local(model.output[0], model.output[1]);

  // logger("output [ %f %f ]", model.output[0], model.output[1]);
  // FIXME: using osc actuation instead
  // ic.actTask.drive(model.output[0], model.output[1], ic.cmdTask.motorScale);
  unsigned long loop_end = micros();
  unsigned long loop_time = loop_end - loop_start;
  if (loop_time > TIMESTEP * 1e6) {
    logger.error("loop running behind (took %4.3fms)", loop_time/1.0e3);
  }

  while (micros() - loop_start < (unsigned long)(TIMESTEP * 1e6));

  // logger("input [%6.4f, %6.4f, %6.4f, %6.4f, %6.4f], output [%6.4f %6.4f] (%3d %3d)",
  //        model.input[0],
  //        model.input[1],
  //        model.input[2],
  //        model.input[3],
  //        model.input[4],
  //        model.output[0],
  //        model.output[1],
  //        act1,
  //        act2);

  // logger("heading %4.2f, roll %4.2f, pitch %4.2f", eul.x(), eul.y(), eul.z());
  // logger("gyro  [ %4.2f,      %4.2f,       %4.2f ]", gyro.z(), gyro.y(), gyro.x());
  // logger(
  //   "crsf %2.4fms, imu %2.4fms, model %2.4fms, drive %2.4fms",
  //   (model_start - loop_start)/1e3f,
  //   (drive_start - model_start)/1e3f,
  //   (loop_end - drive_start)/1e3f
  // );
  // logger("time budget %2.4fms", (TIMESTEP*1e6 - (loop_end - loop_start))/1e3f);
}
