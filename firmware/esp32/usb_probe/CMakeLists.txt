cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# Build only this diagnostic and its declared dependencies. PlatformIO names
# the application component "src"; the IDF default includes every component.
set(COMPONENTS src __pio_env)
project(chong_usb_probe)
