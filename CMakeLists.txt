cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(TensorRTYoloV3 CXX C)


# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set -- defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()
#set the dependcy libs
option(TensorRT-ROOT "if build the excutable bins" /opt/TensorRT-6.0.1.5)
option(OpenCV-ROOT "if build the excutable bins" /home/opencv-4.0.0/build/out)
option(CUDA-ROOT "path to the cuda install" /usr/local/cuda)
include(cmake/ProcessDependency.cmake)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnarrowing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpermissive -std=gnu99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdeprecated-declarations")

set(TensorRTYOLOV3_SRCS)

set(TensorRTYOLOV3_HDRS)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../common)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND TensorRTYOLOV3_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/sampleYoloV3.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/getOptions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/logger.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/sampleEngines.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/sampleOptions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/yoloPlugin.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu"
)

set(YoloV3SRCS)
list(APPEND YoloV3SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/yoloPlugin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu"
    )
CUDA_ADD_EXECUTABLE(TensorRTYOLOV3 ${TensorRTYOLOV3_SRCS})
target_link_libraries(TensorRTYOLOV3  ${TensorRTSSD_DEPENDENCY_LIBS})

CUDA_ADD_LIBRARY(yoloPlugin SHARED ${YoloV3SRCS})
target_link_libraries(yoloPlugin  ${TensorRTSSD_DEPENDENCY_LIBS})
