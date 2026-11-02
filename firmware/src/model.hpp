#pragma once
#include "dl_model_base.hpp"
#include "esp_log.h"
#include "lib/log.h"

#define MINPUT "obs"
#define MOUTPUT "act"

class MLModel {
  dl::Model *model;
	Log log{"model", ANSI_ORANGE};
	bool tested = false;

	std::map<std::string, dl::TensorBase *> modelInputs;
	std::map<std::string, dl::TensorBase *> modelOutputs;

  dl::TensorBase *inputTensor;
  dl::TensorBase *outputTensor;
	
	public:
		float input[9];
		float output[2];

		bool setup() {
			log("setup");
	 		model = new dl::Model("spiffs", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);

			// currently one input: 'input' and one output 'linear_2'
  		modelInputs = model->get_inputs();
  		modelOutputs = model->get_outputs();

			bool keyError = false;
			if (modelInputs.count(MINPUT) == 0) {
				keyError = true;
				log.error("static input key %s not present in model", MINPUT);
			}
			if (modelOutputs.count(MOUTPUT) == 0) {
				keyError = true;
				log.error("static output key %s not present in model", MOUTPUT);
			}

			if (keyError) {
  			for (auto const& [key, tensor] : modelInputs) {
    			log("  input: %s", key.c_str());
  					for (auto const &_val : tensor->get_shape()) {
							log("    %d", _val);
						}
  			}
  			for (auto const& [key, tensor] : modelOutputs) {
    			log("  output: %s", key.c_str());
  					for (auto const &_val : tensor->get_shape()) {
							log("    %d", _val);
						}
  			}
				return false;
			}

			if (std::size(input) != modelInputs[MINPUT]->get_shape()[0]) {
				log.error("static input array length (%d) does not match model's input length (%d)", std::size(input), modelInputs["input"]->get_shape()[0]);
				return false;
			}
			if (std::size(output) != modelOutputs[MOUTPUT]->get_shape()[0]) {
				log.error("static output array length (%d) does not match model's output length (%d)", std::size(output), modelOutputs["output"]->get_shape()[0]);
				return false;
			}

			inputTensor = new dl::TensorBase(modelInputs[MINPUT]->get_shape(), input, 0, dl::DATA_TYPE_FLOAT, false);
			outputTensor = new dl::TensorBase(modelOutputs[MOUTPUT]->get_shape(), output, 0, dl::DATA_TYPE_FLOAT, false);

			return true;
		};

		void test() {
			if (tested) {
				return;
			}

			log("test %s", model->test() == ESP_OK ? "ok" : "err");
			model->profile();
			tested = true;
		}

		void run() {
			modelInputs[MINPUT]->assign(inputTensor);
			model->run();
			outputTensor->assign(modelOutputs[MOUTPUT]);
		};
};
