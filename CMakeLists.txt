cmake_minimum_required(VERSION 3.30)

project(learning_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIREDD ON)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

link_libraries(-lstdc++exp ${VULKAN_LIBRARIES} glfw)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${Vulkan_INCLUDE_DIR}")