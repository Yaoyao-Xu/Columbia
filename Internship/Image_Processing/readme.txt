include_directories:将指定目录添加到编译器的头文件搜索路径之下



指定变量：
CMAKE_CURRENT_SOURCE_DIR：路径指向当前正在处理的源目录（CMakeLists.txt所在目录）
CMAKE_C_COMPILER：指定C编译器
CMAKE_CXX_COMPILER：
CMAKE_C_FLAGS：编译C文件时的选项，如-g；也可以通过add_definitions添加编译选项
EXECUTABLE_OUTPUT_PATH：可执行文件的存放路径
LIBRARY_OUTPUT_PATH：库文件路径
CMAKE_BUILD_TYPE：build 类型(Debug, Release, …)，CMAKE_BUILD_TYPE=Debug
BUILD_SHARED_LIBS：Switch between shared and static libraries
