cmake_minimum_required(VERSION 3.5)
# Enable C++11

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 这样将支持gdb调试
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Define project name
project(SplitObj)

#设置可执行文件与链接库的保存路径
set(EXECUTABLE_OUTPUT_PATH${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH${PROJECT_SOURCE_DIR}/lib)

 
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI


find_package(OpenCV REQUIRED)
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/iouoptimal/
    ${PROJECT_SOURCE_DIR}/stability/
    ${PROJECT_SOURCE_DIR}/FrameDiff/

    )

message(STATUS "OpenCV dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_libs:${OpenCV_LIBS}")

# 假如有src文件夹的话，则一次性将源文件存入 SRC_LIST这个变量里面，方便add_executable,但是不能查找子目录
aux_source_directory(FrameDiff CPP_FILES)
aux_source_directory(iouoptimal IOU_CPP_FILES )
aux_source_directory(stability Motion_CPP)




add_executable(SplitObj main.cpp ${CPP_FILES} ${IOU_CPP_FILES} ${Motion_CPP})

# tianjia opencv lib path
target_link_libraries(SplitObj ${OpenCV_LIBS})
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

 

