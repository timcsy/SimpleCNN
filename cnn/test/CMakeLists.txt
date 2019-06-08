include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${HELLO_BINARY_DIR}/src)
add_executable(test test.cpp)
target_link_libraries(test cnn)