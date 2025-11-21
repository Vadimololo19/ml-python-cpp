#include <drogon/drogon.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <mutex>
#include <unistd.h>
#include <limits.h>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

std::unique_ptr<Ort::Session> g_session;
Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "nids_service_final"};
std::vector<std::string> g_feature_names;
std::atomic<uint64_t> g_request_count{0};
double g_total_inference_time = 0.0; 
std::mutex g_stats_mutex; 

std::string get_executable_path() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return count > 0 ? std::string(result, count) : "";
}

void load_metadata(const std::string& metadata_path) {
    std::ifstream metadata_file(metadata_path);
    if (!metadata_file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл метаданных: " + metadata_path);
    }
    
    json metadata;
    metadata_file >> metadata;
    
    if (!metadata.contains("feature_names") || !metadata["feature_names"].is_array()) {
        throw std::runtime_error("Некорректный формат файла метаданных: отсутствует feature_names");
    }
    
    g_feature_names = metadata["feature_names"].get<std::vector<std::string>>();
    
    spdlog::info("Загружены метаданные из {}", metadata_path);
    spdlog::info("- Всего признаков: {}", g_feature_names.size());
    
    if (!g_feature_names.empty()) {
        std::vector<std::string> first_five;
        size_t count = std::min(size_t(5), g_feature_names.size());
        for (size_t i = 0; i < count; ++i) {
            first_five.push_back(g_feature_names[i]);
        }
        std::string features_str = "{";
        for (size_t i = 0; i < first_five.size(); ++i) {
            features_str += "\"" + first_five[i] + "\"";
            if (i < first_five.size() - 1) features_str += ", ";
        }
        features_str += "}";
        spdlog::debug("Первые {} признаков: {}", count, features_str);
    }
}

float extract_numeric_value(const json& value) {
    if (value.is_number()) {
        return value.get<float>();
    } else if (value.is_string()) {
        std::string str_val = value.get<std::string>();
        str_val.erase(std::remove(str_val.begin(), str_val.end(), ','), str_val.end());
        str_val.erase(std::remove(str_val.begin(), str_val.end(), ' '), str_val.end());
        
        try {
            return std::stof(str_val);
        } catch (...) {
            return 0.0f;
        }
    } else if (value.is_boolean()) {
        return value.get<bool>() ? 1.0f : 0.0f;
    }
    return 0.0f;
}

void predictHandler(const drogon::HttpRequestPtr& req,
                    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        auto json_body = json::parse(req->body());
        if (!json_body.contains("features") || !json_body["features"].is_object()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(R"({"error": "Требуется {'features': {<имена>: <значения>}}"})");
            callback(resp);
            return;
        }

        auto features = json_body["features"];
        std::vector<float> input_values;
        input_values.reserve(g_feature_names.size());
        
        for (const auto& feature_name : g_feature_names) {
            float value = 0.0f;
            
            if (features.contains(feature_name)) {
                value = extract_numeric_value(features[feature_name]);
            } else {
                size_t pos = feature_name.find('_');
                if (pos != std::string::npos) {
                    std::string base_name = feature_name.substr(0, pos);
                    if (features.contains(base_name)) {
                        value = extract_numeric_value(features[base_name]);
                    }
                }
            }
            
            input_values.push_back(value);
        }

        if (input_values.size() != g_feature_names.size()) {
            throw std::runtime_error(
                "Несоответствие размеров: " + 
                std::to_string(input_values.size()) + " вместо " + 
                std::to_string(g_feature_names.size())
            );
        }

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_values.size())};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {"float_input"};
        const char* output_names[] = {"probabilities"};
        
        Ort::RunOptions run_options;
        std::vector<Ort::Value> output_tensors;
        
        try {
            output_tensors = g_session->Run(
                run_options,
                input_names, &input_tensor, 1,
                output_names, 1
            );
        } catch (const Ort::Exception& ex) {
            spdlog::warn("Ошибка с output_names 'probabilities', пробуем 'variable'. Ошибка: {}", ex.what());
            const char* fallback_output_names[] = {"variable"};
            output_tensors = g_session->Run(
                run_options,
                input_names, &input_tensor, 1,
                fallback_output_names, 1
            );
        }

        float attack_prob = 0.0f;
        auto& output_tensor = output_tensors[0];
        auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        if (output_tensor.IsTensor()) {
            if (shape.size() == 2 && shape[1] == 2) {
                auto output_data = output_tensor.GetTensorData<float>();
                attack_prob = output_data[1]; 
            } else if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
                attack_prob = *output_tensor.GetTensorData<float>();
                if (attack_prob > 1.0f) {
                    attack_prob = 1.0f / (1.0f + std::exp(-attack_prob));
                } else if (attack_prob < 0.0f) {
                    attack_prob = 0.0f;
                }
            } else {
                throw std::runtime_error("Неподдерживаемый формат выходного тензора: shape=" + 
                    std::to_string(shape.size()) + " dimensions");
            }
        } else {
            throw std::runtime_error("Выходной тензор не является тензором");
        }

        static std::atomic<uint64_t> request_counter{0};
        uint64_t current_request = ++request_counter;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto inference_duration = end_time - start_time;
        double inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
            inference_duration
        ).count() / 1000.0;  

        if (current_request % 10 == 0) {
          std::string result_str = attack_prob > 0.5f ? "АТАКА" : "НОРМА";
          spdlog::info("[Запрос #{:05d}] {} (вероятность: {:.4f}, время: {:.2f}мс)", 
                 current_request, result_str, attack_prob, inference_time);
        }


        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            g_request_count++;
            g_total_inference_time += inference_time;
        }

        json response = {
            {"prediction", attack_prob},
            {"is_attack", attack_prob > 0.5f},
            {"inference_time_ms", inference_time},
            {"model_version", "rf_nids_csv_v4_cpp_final"},
            {"features_used", static_cast<int>(g_feature_names.size())}
        };

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);

    } catch (const std::exception& e) {
        spdlog::error("Ошибка инференса: {}", e.what());
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setBody(json{{"error", std::string(e.what())}, {"model_version", "rf_nids_csv_v4_cpp_final"}}.dump());
        callback(resp);
    }
}

void healthHandler(const drogon::HttpRequestPtr& /*req*/,
                   std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    uint64_t request_count;
    double total_time;
    
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        request_count = g_request_count.load(std::memory_order_relaxed);
        total_time = g_total_inference_time;
    }
    
    double avg_time = 0.0;
    if (request_count > 0) {
        avg_time = total_time / request_count;
    }
    
    json response = {
        {"status", "ok"},
        {"model", "ONNX NIDS (CSV) - FINAL"},
        {"requests_processed", request_count},
        {"avg_inference_time_ms", avg_time},
        {"feature_count", static_cast<int>(g_feature_names.size())},
        {"model_version", "rf_nids_csv_v4_cpp_final"}
    };
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(response.dump());
    callback(resp);
}

int main() {
    try {
        std::string exe_path = get_executable_path();
        std::string base_dir = fs::path(exe_path).parent_path().string();
        std::string metadata_path = base_dir + "/../../models/metadata.json";
        std::string model_path = base_dir + "/../../models/rf_nids_csv.onnx";
        
        spdlog::info("Базовая директория: {}", base_dir);
        spdlog::info("Путь к метаданным: {}", metadata_path);
        spdlog::info("Путь к модели: {}", model_path);

        try {
            metadata_path = fs::canonical(fs::path(metadata_path)).string();
            model_path = fs::canonical(fs::path(model_path)).string();
        } catch (const fs::filesystem_error& e) {
            spdlog::warn("Ошибка нормализации путей: {}", e.what());
        }

        if (!fs::exists(metadata_path)) {
            throw std::runtime_error("Файл метаданных не найден: " + metadata_path);
        }
        if (!fs::exists(model_path)) {
            throw std::runtime_error("ONNX модель не найдена: " + model_path);
        }

        load_metadata(metadata_path);

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        g_session = std::make_unique<Ort::Session>(
            g_env, 
            model_path.c_str(),
            session_options
        );
        spdlog::info("ONNX модель успешно загружена из {}", model_path);

        drogon::app().registerHandler("/predict", &predictHandler, {drogon::Post});
        drogon::app().registerHandler("/health", &healthHandler, {drogon::Get});

        auto thread_num = std::max(2u, std::thread::hardware_concurrency());
        drogon::app().setThreadNum(thread_num);
        spdlog::info("Используется {} потоков", thread_num);
        
        drogon::app().setLogLevel(trantor::Logger::kInfo);
        drogon::app().addListener("127.0.0.1", 5001);
        
        spdlog::info("Сервер запущен на http://127.0.0.1:5001");
        spdlog::info("Доступные эндпоинты:");
        spdlog::info("  POST /predict - предсказание");
        spdlog::info("  GET  /health  - проверка состояния");
        drogon::app().run();
    } catch (const std::exception& e) {
        spdlog::critical("Критическая ошибка: {}", e.what());
        return 1;
    }
}
