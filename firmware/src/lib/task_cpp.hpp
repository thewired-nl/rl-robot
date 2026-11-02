#pragma once

// from https://forums.freertos.org/t/is-it-possible-create-freertos-task-in-c/2091/2

#include "FreeRTOS.h"
#include "task.h"

class TaskBase {
 public:
  xTaskHandle handle;
};

// class Task : public TaskBase {
//  public:
//   Task(const char* name, void (*taskfun)(void*), unsigned portBASE_TYPE priority, portSHORT stackDepth = configMINIMAL_STACK_SIZE) {
//     xTaskCreate(taskfun, name, stackDepth, this, priority, &handle);
//   }
// };

class TaskClass : public TaskBase {
 public:
  TaskClass(const char* name, unsigned portBASE_TYPE priority, portSHORT stackDepth = configMINIMAL_STACK_SIZE, BaseType_t coreID = 1) {
		Serial.printf("creating task %s, priority %d, stackDepth %d\r\n", name, priority, stackDepth);
		delay(200);
    xTaskCreatePinnedToCore(&taskfun, name, stackDepth, this, priority, &handle, coreID);
  }
  virtual void task() {
		Serial.println("wrong task func called");
	};
  static void taskfun(void* parm) {
		TaskClass *c = static_cast<TaskClass*>(parm);
		c->task();
		// Serial.printf("%p\r\n", c->task);
    // static_cast<TaskClass*>(parm)->task();
    while (1) vTaskDelay(10000);
  }
};
