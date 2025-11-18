cmake_minimum_required(VERSION 3.14)
project(MLServiceCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Drogon CONFIG REQUIRED)

find_package(onnxruntime REQUIRED)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(ml_service_cpp
    main.cpp
)

target_link_libraries(ml_service_cpp
    PRIVATE
        Drogon::Drogon
        onnxruntime::onnxruntime
        nlohmann_json::nlohmann_json
)

target_compile_options(ml_service_cpp PRIVATE
    -O3 -march=native -fno-omit-frame-pointer
)

