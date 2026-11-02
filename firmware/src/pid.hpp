#pragma once
#include <PID_v1.h>

#include <functional>

#include "OSCMessage.h"
#include "inputs.hpp"
#include "lib/log.h"
#include "lib/osc.hpp"

using namespace std::placeholders;

class PIDWrapper {
 public:
  double Setpoint, Input, Output;
  PID pid;

  PIDWrapper(double kp, double ki, double kd) : pid{&Input, &Output, &Setpoint, kp, ki, kd, DIRECT} {
    Setpoint = 0;
    pid.SetMode(AUTOMATIC);
    pid.SetOutputLimits(-1, 1);
  }
};

class PIDController {
  Log log{"pid", ANSI_RED};

  // double Setpoint, Input, Output;
  // PID pid{&Input, &Output, &Setpoint, 1, 0, 0, DIRECT};
  PIDWrapper pidBalance{0.2, 0, 0};
  PIDWrapper pidM1{3, 8, 0};
  PIDWrapper pidM2{3, 8, 0};

  IPAddress _ip{192, 168, 1, 142};

 public:
  float input[9];
  float output[2];

  bool setup() {
    log("setup");

    OSC::addRoute("/pidBalance", [&](OSCMessage& msg, IPAddress ip) {
      _ip = ip;
      float kp = msg.getFloat(0);
      float ki = msg.getFloat(1);
      float kd = msg.getFloat(2);
      // this->pidBalance.Setpoint = -msg.getFloat(3);
      log("balance: updated params kd %4.2f ki %4.2f kd %4.2f", kp, ki, kd);
      this->pidBalance.pid.SetTunings(kp, ki, kd);
    });

    OSC::addRoute("/pidMotors", [&](OSCMessage& msg, IPAddress ip) {
      _ip = ip;
      float kp = msg.getFloat(0);
      float ki = msg.getFloat(1);
      float kd = msg.getFloat(2);
      // this->pidM1.Setpoint = -msg.getFloat(3);
      // this->pidM2.Setpoint = -msg.getFloat(3);
      log("motors: updated params kd %4.2f ki %4.2f kd %4.2f", kp, ki, kd);
      this->pidM1.pid.SetTunings(kp, ki, kd);
      this->pidM2.pid.SetTunings(kp, ki, kd);
    });

    OSC::addRoute("/m1", [&](OSCMessage& msg, IPAddress ip) {
      _ip = ip;
      pidM1.Setpoint = msg.getFloat(0);
    });

    OSC::addRoute("/m2", [&](OSCMessage& msg, IPAddress ip) {
      _ip = ip;
      pidM2.Setpoint = msg.getFloat(0);
    });

    pidBalance.pid.SetSampleTime(5);
    pidM1.pid.SetSampleTime(5);  // ms
    pidM2.pid.SetSampleTime(5);  // ms

    return true;
  };

  void run() {
    pidBalance.Input = input[0];
    if (abs(input[0]) >= 0.55) {
      if (pidBalance.pid.GetMode() == AUTOMATIC) {
        log("i've fallen and i can't get up");
        pidBalance.pid.SetMode(MANUAL);
        pidBalance.Output = 0;
        pidM1.Setpoint = pidBalance.Output;
        pidM2.Setpoint = pidBalance.Output;
      }
    } else {
      pidBalance.pid.SetMode(AUTOMATIC);
      pidBalance.pid.Compute();
      pidM1.Setpoint = -pidBalance.Output;
      pidM2.Setpoint = -pidBalance.Output;
    }

    pidM1.Input = Encoder::normalize(input[3]);
    pidM1.pid.Compute();
    pidM2.Input = Encoder::normalize(input[4]);
    pidM2.pid.Compute();

    OSCMessage msg("/pid_motor");
    msg.add((float)pidM1.Setpoint);
    msg.add((float)pidM1.Input);
    msg.add((float)pidM1.Output);
    msg.add((float)pidM2.Setpoint);
    msg.add((float)pidM2.Input);
    msg.add((float)pidM2.Output);
    OSC::sendMessage(msg, _ip);

    msg.empty();
    msg.setAddress("/balance");
    msg.add((float)pidBalance.Input);
    msg.add((float)pidBalance.Output);
    msg.add((float)pidM1.Input);
    msg.add((float)pidM1.Output);
    OSC::sendMessage(msg, _ip);

    output[0] = pidM1.Output;
    output[1] = pidM2.Output;
    // output[0] = -pidBalance.Output;
    // output[1] = pidBalance.Output;
  };
};
