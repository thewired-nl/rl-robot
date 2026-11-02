Flash model to data/spiffs partition:
`pio pkg exec -p "tool-esptoolpy" -- esptool -c esp32s3 -p /dev/ttyACM0 -b 460800 --before default_reset --after hard_reset write_flash -z --flash-mode dio --flash-size detect 2162688 data/model.espdl`
