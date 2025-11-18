cmake_minimum_required(VERSION 3.14)
project(ml_nids_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR}/.. ABSOLUTE)
message(STATUS "Базовая директория проекта: ${PROJECT_ROOT_DIR}")

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(Drogon CONFIG REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

add_executable(ml_nids_cpp ml_nids.cpp)

target_link_libraries(ml_nids_cpp PRIVATE
    Drogon::Drogon
    onnxruntime::onnxruntime
    nlohmann_json::nlohmann_json
    spdlog::spdlog_header_only
    Threads::Threads
    stdc++fs  
)

target_include_directories(ml_nids_cpp PRIVATE
    ${PROJECT_ROOT_DIR}/models
    ${PROJECT_ROOT_DIR}/load_test
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(ml_nids_cpp PRIVATE
        -O3
        -march=native
        -flto
        -fno-omit-frame-pointer
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
        -fstack-protector-strong
    )
    target_link_options(ml_nids_cpp PRIVATE -flto -Wl,-z,relro,-z,now)
endif()

set_target_properties(ml_nids_cpp PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

macro(check_and_copy_file source_file target_dir)
    if(EXISTS ${source_file})
        add_custom_command(TARGET ml_nids_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${source_file}
                ${target_dir}/
            COMMENT "Копирование ${source_file} в ${target_dir}"
        )
    else()
        message(WARNING "Файл не найден: ${source_file}. Пропускаем копирование.")
    endif()
endmacro()

# Создание директорий и копирование артефактов
add_custom_command(TARGET ml_nids_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:ml_nids_cpp>/models
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:ml_nids_cpp>/load_test
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_ROOT_DIR}/models/metadata.json
        $<TARGET_FILE_DIR:ml_nids_cpp>/models/
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_ROOT_DIR}/models/rf_nids_csv.onnx
        $<TARGET_FILE_DIR:ml_nids_cpp>/models/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_ROOT_DIR}/models/rf_nids_csv.pkl
        $<TARGET_FILE_DIR:ml_nids_cpp>/models/
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_ROOT_DIR}/load_test/payload_nids.json
        $<TARGET_FILE_DIR:ml_nids_cpp>/load_test/
    
    COMMENT "Копирование артефактов обучения в директорию сборки"
)

add_custom_target(show_run_instructions ALL
    COMMAND echo "================================================================================"
    COMMAND echo "C++ сервис успешно собран!"
    COMMAND echo "Для запуска выполните:"
    COMMAND echo "cd ${CMAKE_BINARY_DIR}"
    COMMAND echo "./ml_nids_cpp"
    COMMAND echo ""
    COMMAND echo "Сервис будет доступен на http://127.0.0.1:5001"
    COMMAND echo "Эндпоинты:"
    COMMAND echo "POST /predict - предсказание атаки"
    COMMAND echo "GET  /health  - проверка состояния сервиса"
    COMMAND echo ""
    COMMAND echo "Пути к артефактам:"
    COMMAND echo "Модели: ${PROJECT_ROOT_DIR}/models/"
    COMMAND echo "Тестовые данные: ${PROJECT_ROOT_DIR}/load_test/"
    COMMAND echo "================================================================================"
    VERBATIM
)

add_dependencies(show_run_instructions ml_nids_cpp)

message(STATUS "Drogon версия: ${Drogon_VERSION}")
message(STATUS "ONNXRuntime версия: ${ONNXRuntime_VERSION}")
message(STATUS "nlohmann_json версия: ${nlohmann_json_VERSION}")
message(STATUS "spdlog версия: ${spdlog_VERSION}")
