cmake_minimum_required(VERSION 3.11)
project(rafeeq_speech)

find_package(ament_cmake REQUIRED)

# Install the ROS 2 Python node as an executable
install(PROGRAMS
  scripts/speech_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install model files (tflite + labels)
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
