#pragma once
#include <FreeRTOS.h>
#include <HardwareSerial.h>
#include <Wire.h>

#include <span>

#include "driver/pulse_cnt.h"
#include "lib/log.h"
#include "lib/osc.hpp"
#include "lib/task_cpp.hpp"

#define INPUT_SIZE 5

using InputUpdate = float[INPUT_SIZE];

class InputClass {
 public:
  QueueHandle_t queue;
  InputUpdate input;
  uint8_t size;
  bool started = false;
  unsigned long lastUpdate;

  InputClass(uint8_t s) : size(s) {
    queue = xQueueCreate(1, size * sizeof(float));
  }

  virtual const char* name() = 0;

  bool setup() {
    started = _setup();
    return started;
  }

  virtual bool _setup() = 0;

  void qUpdate() {
    xQueueOverwrite(queue, input);
  }
};

// ICM45686
#include "ICM45686.h"
#include "vendor/KalmanFilter.hpp"
class IMUTask : public TaskClass, public InputClass {
  typedef struct vec3 {
    float x, y, z;
  } vec3;

  Log log{"inputs:imu", ANSI_ORANGE};

  ICM456xx IMU{Wire, 0};

  inv_imu_sensor_data_t imu_data;
  vec3 accel_g = {0};
  vec3 gyro_rps = {0};

  KalmanFilter kalmanPitch{0.001, 0.003, 0.03};
  float alpha = 0.99f;

  float pitch;

  static constexpr const int ACCEL_FSR_G = 16;
  static constexpr const int GYRO_FSR_DPS = 2000;  // 35 rad/s
  static constexpr const float MAX_LSB = 32768.0f;

  static constexpr const float PITCH_OFFSET = 0.025f;

 public:
  static constexpr const char* _name = "imu";
  IMUTask(uint8_t size) : TaskClass("IMU", 2, 20000), InputClass(size) {}

  const char* name() {
    return _name;
  }

  void sendData(IPAddress ip) {
    OSCMessage resp("/imu_calib");
    resp.add(accel_g.x);
    resp.add(accel_g.y);
    resp.add(accel_g.z);
    resp.add(gyro_rps.x);
    resp.add(gyro_rps.y);
    resp.add(gyro_rps.z);
    OSC::sendMessage(resp, ip);
  }

  bool _setup() {
    if (IMU.begin() != 0) {
      log.error("could not init ICM45686");
      return false;
    }

    // 200Hz ODR = once every 5ms
    IMU.startAccel(400, ACCEL_FSR_G);
    IMU.startGyro(400, GYRO_FSR_DPS);

    // IMU.startAccel(1000, ACCEL_FSR_G);
    // IMU.startGyro(1000, GYRO_FSR_DPS);

    // OSC::addRoute("/imu", [&](OSCMessage& msg, IPAddress ip) {
    //   int samples = msg.getInt(0);

    //   accel_g = {0, 0, 0};
    //   gyro_rps = {0, 0, 0};

    //   sendData(ip);

    //   for (int i = 0; i < samples; i++) {
    //     IMU.getDataFromRegisters(imu_data);

    //     accel_g.x += (imu_data.accel_data[0] * ACCEL_FSR_G) / MAX_LSB;  // right
    //     accel_g.y += (imu_data.accel_data[1] * ACCEL_FSR_G) / MAX_LSB;  // forwards
    //     accel_g.z += (imu_data.accel_data[2] * ACCEL_FSR_G) / MAX_LSB;  // up

    //     gyro_rps.x += (imu_data.gyro_data[0] * GYRO_FSR_DPS) / MAX_LSB;  // pitch
    //     gyro_rps.y += (imu_data.gyro_data[1] * GYRO_FSR_DPS) / MAX_LSB;  // roll
    //     gyro_rps.z += (imu_data.gyro_data[2] * GYRO_FSR_DPS) / MAX_LSB;  // yaw
    //   }

    //   accel_g.x /= samples / 1000.0f;  // mg
    //   accel_g.y /= samples / 1000.0f;  // mg
    //   accel_g.z /= samples / 1000.0f;  // mg

    //   gyro_rps.x /= samples;  // deg
    //   gyro_rps.y /= samples;  // deg
    //   gyro_rps.z /= samples;  // deg
    //   log("%d  accel: %f, %f, %f  gyro: %f, %f, %f", samples, accel_g.x, accel_g.y, accel_g.z, gyro_rps.x, gyro_rps.y, gyro_rps.z);

    //   sendData(ip);
    // });

    return true;
  }

  void task() {
    while (!started) {
      vTaskDelay(2);
    }

    vTaskDelay(100);

    for (;;) {
      unsigned long loop_start = micros();

      int stat = IMU.getDataFromRegisters(imu_data);
      if (stat != INV_IMU_OK) {
        if (stat == INV_IMU_ERROR) {
          log.error("unspecified");
        } else if (stat == INV_IMU_ERROR_TRANSPORT) {
          log.error("transport");
        } else if (stat == INV_IMU_ERROR_TIMEOUT) {
          log.error("timeout");
        } else if (stat == INV_IMU_ERROR_BAD_ARG) {
          log.error("bad arg");
        } else if (stat == INV_IMU_ERROR_EDMP_ODR) {
          log.error("edmp odr");
        } else if (stat == INV_IMU_ERROR_EDMP_BUF_EMPTY) {
          log.error("edmp buffer empty");
        } else {
          log.error("unknown");
        }
        vTaskDelay(1);
        continue;
      }

      unsigned long data_end = micros();

      accel_g.x = (imu_data.accel_data[0] * ACCEL_FSR_G) / MAX_LSB;  // right
      accel_g.y = (imu_data.accel_data[1] * ACCEL_FSR_G) / MAX_LSB;  // forwards
      accel_g.z = (imu_data.accel_data[2] * ACCEL_FSR_G) / MAX_LSB;  // up

      gyro_rps.x = (float)(imu_data.gyro_data[0] * GYRO_FSR_DPS) / MAX_LSB * DEG_TO_RAD;  // pitch
      gyro_rps.y = (float)(imu_data.gyro_data[1] * GYRO_FSR_DPS) / MAX_LSB * DEG_TO_RAD;  // roll
      gyro_rps.z = (float)(imu_data.gyro_data[2] * GYRO_FSR_DPS) / MAX_LSB * DEG_TO_RAD;  // yaw

      // float pitch_rate = (float)(imu_data.gyro_data[0] * GYRO_FSR_DPS) / MAX_LSB * (M_PI / 180.0f);
      // gyro_dps[1] = (float)(imu_data.gyro_data[1] * GYRO_FSR_DPS) / MAX_LSB;
      // float yaw_rate = (float)(imu_data.gyro_data[2] * GYRO_FSR_DPS) / MAX_LSB * (M_PI / 180.0f);

      // float accel_pitch = atan2f(accel_fwd, accel_up);
      // float accel_pitch = atan2f(accel_g.y, sqrt(accel_g.x * accel_g.x + accel_g.z * accel_g.z));
      // float accel_pitch = atan2f(accel_g.y, sqrt(accel_g.x * accel_g.x + accel_g.z * accel_g.z));
      float accel_pitch = atan2f(accel_g.y, accel_g.z);

      // Complementary filter mix accel + gyro to get pitch
      pitch = alpha * (pitch + (gyro_rps.x * (data_end - lastUpdate) / 1e6f)) + (1.0f - alpha) * accel_pitch;

      // Kalman filter (breaks sometimes..)
      // pitch = kalmanPitch.update(accel_pitch, gyro_rps.x);

      unsigned long calc_end = micros();

      // [ pitch, pitch_rate, yaw_rate ]
      input[0] = pitch;
      input[1] = gyro_rps.x;
      input[2] = gyro_rps.z;

      // log("pitch %5.3f, accel_pitch %5.3f, pitch_rate %5.3f", pitch, accel_pitch, gyro_rps.x);
      // log("update took %2.4fms: data %2.4fms, calc %2.4fms (%2.4fms since last)",
      //     (calc_end - loop_start) / 1e3f,
      //     (data_end - loop_start) / 1e3f,
      //     (calc_end - data_end) / 1e3f,
      //     (micros() - lastUpdate) / 1e3f);

      lastUpdate = micros();

      // instead of force overwriting queue as fast as possible, wait for previous input to be read out
      xQueueSend(queue, input, portMAX_DELAY);

      vTaskDelay(1);
    }
  }
};

// BNO055
// #include <Adafruit_BNO055.h>
// #include <Adafruit_Sensor.h>
// #include <utility/imumaths.h>
// class IMUTask : public TaskClass, public InputClass {
//   Log log{"inputs:imu", ANSI_ORANGE};

//   Adafruit_BNO055 bno{55, 0x28};

//   // data
//   imu::Quaternion q;
//   imu::Vector<3> eul;
//   imu::Vector<3> gyro;

//  public:
//   static constexpr const char* _name = "imu";
//   IMUTask(uint8_t size) : TaskClass("IMU", 2, 20000), InputClass(size) {}

//   const char* name() {
//     return _name;
//   }

//   bool _setup() {
//     if (!bno.begin()) {
//       log.error("could not init BNO055");
//       return false;
//     }

//     bno.setExtCrystalUse(true);
//     bno.setMode(OPERATION_MODE_IMUPLUS);

//     Adafruit_BNO055::adafruit_bno055_rev_info_t info;
//     bno.getRevInfo(&info);
//     log("bno055 revision %d", info.sw_rev);

//     return true;
//   }

//   void task() {
//     while (!started) {
//       vTaskDelay(2);
//     }

//     vTaskDelay(100);

//     for (;;) {
//       // getting new data from BNO055 takes between 2-5ms
//       unsigned long loop_start = micros();

//       // BNO055 Euler calculation is broken, get Quaternion and let library convert it to Euler
//       q = bno.getQuat();
//       gyro = bno.getVector(bno.VECTOR_GYROSCOPE);

//       unsigned long data_end = micros();

//       q.normalize();
//       eul = q.toEuler();
//       gyro.toRadians();

//       unsigned long calc_end = micros();

//       // [ pitch, pitch_rate, yaw_rate ]
//       input[0] = eul.z();
//       input[1] = gyro.x();
//       input[2] = gyro.z();
//       qUpdate();

//       // log("heading %4.2f, roll %4.2f, pitch %4.2f", eul.x(), eul.y(), eul.z());
//       // log("gyro  [ %4.2f,      %4.2f,       %4.2f ]", gyro.z(), gyro.y(), gyro.x());
//       // log("update took %2.4fms: data %2.4fms, calc %2.4fms (%2.4fms since last)",
//       //     (calc_end - loop_start) / 1e3f,
//       //     (data_end - loop_start) / 1e3f,
//       //     (calc_end - data_end) / 1e3f,
//       //     (micros() - lastUpdate) / 1e3f);

//       lastUpdate = micros();
//       vTaskDelay(1);
//     }
//   }
// };

// 4x quadrature encoder using esp32 PCNT peripheral
class Encoder {
  Log log{"inputs:encoder", ANSI_CYAN};
  pcnt_unit_handle_t pcntUnit;
  pcnt_channel_handle_t pcntChA;
  pcnt_channel_handle_t pcntChB;

  int direction;
  unsigned long lastUpdate = 0;

  static constexpr const float PULSE_PER_ROT = 7 * 4 * 30 * 2;  // 7 ppr with 4x quadrature, 30:1 gearbox + 2:1 printed gears
  static constexpr const float MAX_SPEED = 37;
  // static constexpr const float PULSE_PER_ROT = 7 * 4 * 30;  // 7 ppr with 4x quadrature, 30:1 gearbox + 1:1 printed gears
  // static constexpr const float MAX_SPEED = 66;

  static constexpr const float alpha = 0.0f;

  float _speed_rps;  // rad/s
  float _speed;      // normalized -1.0 to 1.0 by dividing with MAX_SPEED
 public:
  Encoder(int pinA, int pinB, bool reverse = false) : direction(reverse ? -1 : 1) {
    pcnt_unit_config_t unitConfig = {
        .low_limit = INT16_MIN + 1,
        .high_limit = INT16_MAX - 1,
    };
    ESP_ERROR_CHECK(pcnt_new_unit(&unitConfig, &pcntUnit));

    pcnt_glitch_filter_config_t filterConfig = {
        .max_glitch_ns = 1000};
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(pcntUnit, &filterConfig));

    pcnt_chan_config_t chAConfig = {
        .edge_gpio_num = pinA,
        .level_gpio_num = pinB,
    };
    ESP_ERROR_CHECK(pcnt_new_channel(pcntUnit, &chAConfig, &pcntChA));

    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(pcntChA, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(pcntChA, PCNT_CHANNEL_LEVEL_ACTION_INVERSE, PCNT_CHANNEL_LEVEL_ACTION_KEEP));

    pcnt_chan_config_t chBConfig = {
        .edge_gpio_num = pinB,
        .level_gpio_num = pinA,
    };
    ESP_ERROR_CHECK(pcnt_new_channel(pcntUnit, &chBConfig, &pcntChB));

    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(pcntChB, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(pcntChB, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));

    ESP_ERROR_CHECK(pcnt_unit_enable(pcntUnit));
    ESP_ERROR_CHECK(pcnt_unit_start(pcntUnit));
  }

  int count() {
    int raw = 0;
    pcnt_unit_get_count(pcntUnit, &raw);

    pcnt_unit_clear_count(pcntUnit);
    // if (pcnt_unit_start(pcntUnit) != ESP_OK) {
    //   log.error("failed to start pcnt unit");
    // }

    return raw * direction;
  }

  float update() {
    unsigned long now = micros();
    float raw = static_cast<float>(count());
    // log("count %.0f, in %4.2fms", raw, (now - lastUpdate) / 1e3f);

    float time_s = (now - lastUpdate) / 1e6f;
    _speed_rps = alpha * _speed_rps + (1 - alpha) * (raw / PULSE_PER_ROT * 2 * PI / time_s);
    
    lastUpdate = now;
    return _speed_rps;
  }

  static float normalize(float speed_rps) {
    return constrain(speed_rps / MAX_SPEED, -1.0, 1.0);
  }
};

class EncoderTask : public TaskClass, public InputClass {
  Log log{"inputs:encoder", ANSI_CYAN};

  Encoder encoder1{3, 4, false};
  Encoder encoder2{5, 6, false};

 public:
  static constexpr const char* _name = "enc";
  EncoderTask(uint8_t size) : TaskClass("Encoder", 3, 20000), InputClass(size) {}

  const char* name() {
    return _name;
  }

  bool _setup() {
    return true;
  }

  void task() {
    while (!started) {
      vTaskDelay(2);
    }

    vTaskDelay(100);

    int acc = 0;

    for (;;) {
      // int enc1 = encoder1.count();
      // int enc2 = encoder2.count();
      // log("enc 1: %d, enc 2: %d, in %4.2fms", enc1, enc2, (micros() - lastUpdate)/1e3f);
      // input[0] = constrain(encoder1.speed(), -50.0, 50.0);
      // input[1] = constrain(encoder2.speed(), -50.0, 50.0);
      input[0] = encoder1.update();
      input[1] = encoder2.update();
      // log("enc1 %4.2f enc2 %4.2f", input[0], input[1]);
      // lastUpdate = micros();

      // acc += encoder1.count();
      // log("count enc1 %d", acc);
      // vTaskDelay(100);

      // instead of force overwriting queue as fast as possible, wait for previous input to be read out
      xQueueSend(queue, input, portMAX_DELAY);
      // vTaskDelay(10);
    };
  }
};

#include <AlfredoCRSF.h>
class CMDTask : public TaskClass, public InputClass {
  Log log{"inputs:cmd", ANSI_GREEN};

  HardwareSerial crsfSerial;
  AlfredoCRSF crsf;

 public:
  static constexpr const char* _name = "cmd";
  CMDTask(uint8_t size) : TaskClass("CMD", 2, 20000), InputClass(size), crsfSerial(1) {}

  const char* name() {
    return _name;
  }

  bool _setup() {
    crsfSerial.begin(CRSF_BAUDRATE, SERIAL_8N1, 1, 2);  // on receiver: 1 TX, 2 RX
    if (!crsfSerial) {
      log.error("Invalid CRSF Serial config");
      return false;
    }

    crsf.begin(crsfSerial);

    log("crsf receiver setup");

    return true;
  }

  float motorScale = 1.0;
  bool switchA = true;

  void task() {
    while (!started) {
      vTaskDelay(2);
    }

    vTaskDelay(100);

    for (;;) {
      crsf.update();
      if (crsf.isLinkUp()) {
        motorScale = constrain((crsf.getChannel(3) - 1000) / 1000.0, 0.0, 1.0);  // throttle
        // pitch = constrain((crsf.getChannel(2)-1500)/1000.0, -1.0, 1.0); // pitch
        bool switchAUpd = crsf.getChannel(5) > 1500;  // switch between model/manual
        if (switchAUpd != switchA) {
          log("switchA toggle: %s", switchAUpd ? "on" : "off");
          switchA = switchAUpd;
        }

        // inputs [cmd_vel, cmd_yaw]
        input[0] = constrain((crsf.getChannel(2) - 1500) / 500.0, -1.0, 1.0);  // velocity (pitch)
        input[1] = constrain((crsf.getChannel(1) - 1500) / 500.0 * -1.0, -1.0, 1.0);  // yaw

        for (int i = 0; i < 2; i++) {
          if (abs(input[i]) < 0.08) {
            input[i] = 0;
          }
        }

        qUpdate();

        // for (int i = 4; i <= 12; i++) {
        //   log("  channel %d: %d", i, crsf.getChannel(i));
        // }
      } else {
        input[0] = 0.0;
        input[1] = 0.0;
        qUpdate();
      }
      // input[0] = 0.0;
      // input[1] = 0.0;
      // xQueueSend(queue, input, portMAX_DELAY);
      // qUpdate();

      // log("%2.4fms since last update", (micros() - lastUpdate) / 1e3f);

      vTaskDelay(1);
    }
  }
};

#include "pid.hpp"

class ActuatorTask : public TaskClass, public InputClass {
  Log log{"inputs:act", ANSI_GREEN};

// MotorA: right
#define MotorA1 7
#define MotorA2 8
#define MotorApwm 11
#define InvertMotorA false

// MotorB: left
#define MotorB1 9
#define MotorB2 10
#define MotorBpwm 12
#define InvertMotorB true

  PIDWrapper pidMA{3, 8, 0};
  PIDWrapper pidMB{3, 8, 0};

  float _l1;
  float _l2;

  bool useLocal = true;

 public:
  static constexpr const char* _name = "actuators";
  ActuatorTask(uint8_t size) : TaskClass("Actuators", 2, 20000), InputClass(size) {}

  const char* name() {
    return _name;
  }

  bool _setup() {
    analogWriteFrequency(MotorApwm, 500);
    analogWriteResolution(MotorApwm, 10);
    pinMode(MotorA1, OUTPUT);
    pinMode(MotorA2, OUTPUT);
    // analogWriteFrequency(MotorA1, 500);
    // analogWriteFrequency(MotorA2, 500);
    // analogWriteResolution(MotorA1, 10);
    // analogWriteResolution(MotorA2, 10);
    
    analogWriteFrequency(MotorBpwm, 500);
    analogWriteResolution(MotorBpwm, 10);
    pinMode(MotorB1, OUTPUT);
    pinMode(MotorB2, OUTPUT);
    // analogWriteFrequency(MotorB1, 500);
    // analogWriteFrequency(MotorB2, 500);
    // analogWriteResolution(MotorB1, 10);
    // analogWriteResolution(MotorB2, 10);

    pidMA.pid.SetSampleTime(5); // ms
    pidMB.pid.SetSampleTime(5); // ms

    OSC::addRoute("/act", [&](OSCMessage& msg, IPAddress ip) {
      useLocal = false;
      float act1 = msg.getFloat(0);
      float act2 = msg.getFloat(1);
      drive(act1, act2, 1.0);
      log("diff [ %4.2f %4.2f ]", _l1 - act1, _l2 - act2);
    });

    return true;
  }

  void drive(float act1, float act2, float scale) {
    input[0] = constrain(act1, -1.0f, 1.0f);
    input[1] = constrain(act2, -1.0f, 1.0f);
    _drive();
    
    // pidMA.Setpoint = input[0] * scale;
    // pidMB.Setpoint = input[1] * scale;

    qUpdate();
  }

  void encoders(float speed1, float speed2) {
    pidMA.Input = speed1;
    pidMB.Input = speed2;
  }

  void local(float l1, float l2) {
    _l1 = l1;
    _l2 = l2;
    if (useLocal) {
      drive(l1, l2, 1.0);
    }
  }

  void _drive() {
    // motor driver A/B are flipped from actuator 1/2
    int actA = constrain(lround(1024.0f * input[1]), -1024, 1024);
    int actB = constrain(lround(1024.0f * input[0]), -1024, 1024);
    // int actA = constrain(lround(1024.0f * pidMA.Output), -1024, 1024);
    // int actB = constrain(lround(1024.0f * pidMB.Output), -1024, 1024);

    // if (abs(actA) < 0.1) {
    //   actA = 0;
    // }

    // if (abs(actB) < 0.1) {
    //   actB = 0;
    // }

    if (InvertMotorA) {
      actA *= -1;
    }

    if (InvertMotorB) {
      actB *= -1;
    }

    // log("drive inA %4.2f, inB %4.2f, actA %d, actB %d", input[0], input[1], actA, actB);

    if (actA >= 0) {
      analogWrite(MotorApwm, actA);
      digitalWrite(MotorA1, HIGH);
      digitalWrite(MotorA2, LOW);
      // analogWrite(MotorA1, actA);
      // analogWrite(MotorA2, 0);
    } else {
      analogWrite(MotorApwm, -actA);
      digitalWrite(MotorA1, LOW);
      digitalWrite(MotorA2, HIGH);
      // analogWrite(MotorA1, 0);
      // analogWrite(MotorA2, -1 * actA);
    }

    if (actB >= 0) {
      analogWrite(MotorBpwm, actB);
      digitalWrite(MotorB1, HIGH);
      digitalWrite(MotorB2, LOW);
      // analogWrite(MotorB1, actB);
      // analogWrite(MotorB2, 0);
    } else {
      analogWrite(MotorBpwm, -actB);
      digitalWrite(MotorB1, LOW);
      digitalWrite(MotorB2, HIGH);
      // analogWrite(MotorA1, 0);
      // analogWrite(MotorA2, -1 * actA);
    }

    // if (actB >= 0) {
    //   analogWrite(MotorB1, actB);
    //   analogWrite(MotorB2, 0);
    // } else {
    //   analogWrite(MotorB1, 0);
    //   analogWrite(MotorB2, -1 * actB);
    // }
  }

  void task() {
    while (!started) {
      vTaskDelay(2);
    }

    vTaskDelay(100);

    for (;;) {
      // pidMA.pid.Compute();
      // pidMB.pid.Compute();
      // _drive();
      // qUpdate();
      vTaskDelay(portMAX_DELAY);
    }
  }
};

class InputCollector {
  Log log{"inputs", ANSI_PURPLE};

 public:
  IMUTask imuTask{3};
  EncoderTask encTask{2};
  CMDTask cmdTask{2};
  ActuatorTask actTask{2};

  std::vector<InputClass*> inputs;

  InputCollector() {
    inputs.push_back(&imuTask);
    inputs.push_back(&encTask);
    inputs.push_back(&cmdTask);
    inputs.push_back(&actTask);
  }

  bool setup() {
    bool succ = true;
    for (auto i : inputs) {
      succ &= i->setup();
    }
    return succ;
  }

  void update(float* output) {
    uint8_t offset = 0;
    for (auto i : inputs) {
      if (xQueueReceive(i->queue, output + offset, 0) != pdPASS) {
        log.error("  %s failed to provide new data", i->name());
      }
      offset += i->size;
    }
    log("input [ %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f ]", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]);
    // actTask.encoders(Encoder::normalize(output[3]), Encoder::normalize(output[4]));
    // log("input [ %f %f %f %f %f %f %f %f %f ]", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]);
  }
};
