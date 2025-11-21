#include <drogon/drogon.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <unistd.h>
#include <limits.h>

using json = nlohmann::json;
namespace fs = std::filesystem;
std::unique_ptr<Ort::Session> g_session;
Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "nids_service"};
std::vector<std::string> g_feature_names;

std::atomic<uint64_t> g_request_count{0};
double g_total_inference_time = 0.0;
std::mutex g_stats_mutex;

struct PreprocessingParams {
    std::unordered_map<std::string, double> numerical_medians;
    std::unordered_map<std::string, double> numerical_means;
    std::unordered_map<std::string, double> numerical_scales;
    std::unordered_map<std::string, std::vector<std::string>> categorical_categories;
    std::vector<std::string> numerical_features;
    std::vector<std::string> categorical_features;
};

PreprocessingParams g_preprocessing_params;

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
    spdlog::info("Загружено {} признаков из {}", g_feature_names.size(), metadata_path);
}

void load_preprocessing_params(const std::string& params_path) {
    if (!fs::exists(params_path)) {
        spdlog::warn("Файл параметров предобработки не найден: {}", params_path);
        spdlog::warn("Будет использована упрощенная предобработка");
        return;
    }
    
    std::ifstream params_file(params_path);
    if (!params_file.is_open()) {
        spdlog::warn("Не удалось открыть файл параметров предобработки: {}", params_path);
        return;
    }
    
    try {
        json params;
        params_file >> params;
        
        if (params.contains("numerical_features") && params["numerical_features"].is_array()) {
            g_preprocessing_params.numerical_features = params["numerical_features"].get<std::vector<std::string>>();
            spdlog::info("Числовые признаки: {}", g_preprocessing_params.numerical_features.size());
        }
        
        if (params.contains("numerical_medians") && params["numerical_medians"].is_object()) {
            for (auto& [key, value] : params["numerical_medians"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_medians[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено медианных значений: {}", g_preprocessing_params.numerical_medians.size());
        }
        
        if (params.contains("numerical_means") && params["numerical_means"].is_object()) {
            for (auto& [key, value] : params["numerical_means"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_means[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено средних значений: {}", g_preprocessing_params.numerical_means.size());
        }
        
        if (params.contains("numerical_scales") && params["numerical_scales"].is_object()) {
            for (auto& [key, value] : params["numerical_scales"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_scales[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено масштабных коэффициентов: {}", g_preprocessing_params.numerical_scales.size());
        }
        
        if (params.contains("categorical_features") && params["categorical_features"].is_array()) {
            g_preprocessing_params.categorical_features = params["categorical_features"].get<std::vector<std::string>>();
            spdlog::info("Категориальные признаки: {}", g_preprocessing_params.categorical_features.size());
        }
        
        if (params.contains("categorical_categories") && params["categorical_categories"].is_object()) {
            for (auto& [feature_name, categories] : params["categorical_categories"].items()) {
                if (categories.is_array()) {
                    std::vector<std::string> cat_list;
                    for (auto& cat : categories) {
                        if (cat.is_string()) {
                            cat_list.push_back(cat.get<std::string>());
                        }
                    }
                    if (!cat_list.empty()) {
                        g_preprocessing_params.categorical_categories[feature_name] = cat_list;
                    }
                }
            }
            spdlog::info("Загружено категорий для {} признаков", g_preprocessing_params.categorical_categories.size());
        }
        
        spdlog::info("Параметры предобработки успешно загружены");
    } catch (const std::exception& e) {
        spdlog::error("Ошибка при загрузке параметров предобработки: {}", e.what());
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

std::vector<float> preprocess_input_features(const json& features_json) {
    std::vector<float> processed_features(g_feature_names.size(), 0.0f);
    
    for (const auto& feature_name : g_preprocessing_params.numerical_features) {
        float value = 0.0f;
        
        if (features_json.contains(feature_name)) {
            value = extract_numeric_value(features_json[feature_name]);
        } else if (g_preprocessing_params.numerical_medians.find(feature_name) != 
                   g_preprocessing_params.numerical_medians.end()) {
            value = static_cast<float>(g_preprocessing_params.numerical_medians[feature_name]);
            spdlog::debug("Использовано медианное значение для {}: {}", feature_name, value);
        }
        
        if (g_preprocessing_params.numerical_means.find(feature_name) != 
            g_preprocessing_params.numerical_means.end() &&
            g_preprocessing_params.numerical_scales.find(feature_name) != 
            g_preprocessing_params.numerical_scales.end()) {
            
            double mean = g_preprocessing_params.numerical_means[feature_name];
            double scale = g_preprocessing_params.numerical_scales[feature_name];
            
            if (scale > 1e-8) { 
                value = static_cast<float>((value - mean) / scale);
            }
        }
        
        auto it = std::find(g_feature_names.begin(), g_feature_names.end(), feature_name);
        if (it != g_feature_names.end()) {
            size_t idx = std::distance(g_feature_names.begin(), it);
            processed_features[idx] = value;
        }
    }
    
    for (const auto& feature_name : g_preprocessing_params.categorical_features) {
        std::string value = "Missing";
        
        if (features_json.contains(feature_name)) {
            if (features_json[feature_name].is_string()) {
                value = features_json[feature_name].get<std::string>();
            } else {
                try {
                    value = std::to_string(extract_numeric_value(features_json[feature_name]));
                } catch (...) {
                    value = "Missing";
                }
            }
        }
        
        if (g_preprocessing_params.categorical_categories.find(feature_name) != 
            g_preprocessing_params.categorical_categories.end()) {
            
            const auto& categories = g_preprocessing_params.categorical_categories[feature_name];
            for (const auto& category : categories) {
                std::string full_feature_name = feature_name + "_" + category;
                
                auto it = std::find(g_feature_names.begin(), g_feature_names.end(), full_feature_name);
                if (it != g_feature_names.end()) {
                    size_t idx = std::distance(g_feature_names.begin(), it);
                    processed_features[idx] = (value == category) ? 1.0f : 0.0f;
                }
            }
        }
    }
    
    return processed_features;
}

void predictHandler(const drogon::HttpRequestPtr& req,
                    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
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
        if (!g_preprocessing_params.numerical_features.empty() || 
            !g_preprocessing_params.categorical_features.empty()) {
            input_values = preprocess_input_features(features);
        } else {
            input_values.reserve(g_feature_names.size());
            for (const auto& feature_name : g_feature_names) {
                float value = 0.0f;
                if (features.contains(feature_name)) {
                    value = extract_numeric_value(features[feature_name]);
                }
                input_values.push_back(value);
            }
        }

        if (input_values.size() != g_feature_names.size()) {
            spdlog::warn("Несоответствие количества признаков: {} вместо {}", 
                        input_values.size(), g_feature_names.size());
        }

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_values.size())};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

        const char* input_name = "float_input";
        const char* output_names[] = {"probabilities", "variable", "output_label", "output_probability"};
        
        std::vector<Ort::Value> output_tensors;
        Ort::RunOptions run_options;
        float attack_prob = 0.0f;
        bool success = false;
        
        for (const auto* output_name : output_names) {
            try {
                output_tensors = g_session->Run(
                    run_options,
                    &input_name, &input_tensor, 1,
                    &output_name, 1
                );
                
                if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                    auto& output_tensor = output_tensors[0];
                    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
                    auto shape = tensor_info.GetShape();
                    
                    auto output_data = output_tensor.GetTensorData<float>();
                    
                    if (shape.size() == 2 && shape[1] == 2) {
                        attack_prob = output_data[1];
                    } else if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
                        float value = output_data[0];
                        if (value < 0.0f || value > 1.0f) {
                            attack_prob = 1.0f / (1.0f + std::exp(-value));
                        } else {
                            attack_prob = value;
                        }
                    } else if (tensor_info.GetElementCount() >= 2) {
                        attack_prob = output_data[1];
                    }
                    
                    spdlog::debug("Использовано имя выхода '{}', вероятность атаки: {:.4f}", 
                                 output_name, attack_prob);
                    success = true;
                    break;
                }
            } catch (const Ort::Exception& ex) {
                spdlog::debug("Попытка с '{}' не удалась: {}", output_name, ex.what());
                continue;
            }
        }
        
        if (!success) {
            throw std::runtime_error("Не удалось получить корректный вывод от модели");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double inference_time_ms = duration.count() / 1000.0;

        bool is_attack = attack_prob > 0.5f;
        std::string result_str = is_attack ? "АТАКА" : "НОРМА";
        
        {
            uint64_t request_number = ++g_request_count;
            spdlog::info("[Запрос #{:06d}] {} (вероятность: {:.4f}, время: {:.2f}мс)", 
                        request_number, result_str, attack_prob, inference_time_ms);
        }

        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            g_total_inference_time += inference_time_ms;
        }

        json response = {
            {"prediction", attack_prob},
            {"is_attack", is_attack},
            {"inference_time_ms", inference_time_ms},
            {"model_version", "rf_nids_csv_v3_cpp_fixed"},
            {"features_used", static_cast<int>(g_feature_names.size())}
        };

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);

    } catch (const std::exception& e) {
        spdlog::error("Ошибка инференса: {}", e.what());
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setBody(json{{"error", e.what()}, {"model_version", "rf_nids_csv_v3_cpp_fixed"}}.dump());
        callback(resp);
    }
}

void healthHandler(const drogon::HttpRequestPtr& req,
                  std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    uint64_t request_count;
    double total_time;
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        request_count = g_request_count.load();
        total_time = g_total_inference_time;
    }
    
    double avg_time = 0.0;
    if (request_count > 0) {
        avg_time = total_time / request_count;
    }
    
    json response = {
        {"status", "ok"},
        {"model", "ONNX NIDS (CSV)"},
        {"requests_processed", request_count},
        {"avg_inference_time_ms", avg_time},
        {"feature_count", static_cast<int>(g_feature_names.size())},
        {"model_version", "rf_nids_csv_v3_cpp_fixed"}
    };
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(response.dump());
    callback(resp);
}

int main()
{
    try {
        std::string exe_path = get_executable_path();
        std::string base_dir = fs::path(exe_path).parent_path().string();
        
        std::string metadata_path = base_dir + "/../models/metadata.json";
        std::string model_path = base_dir + "/../models/rf_nids_csv.onnx";
        std::string preprocessing_path = base_dir + "/../../models/preprocessing_params.json";
        
        try {
            if (fs::exists(metadata_path)) {
                metadata_path = fs::canonical(metadata_path).string();
            } else {
                std::vector<std::string> alt_paths = {
                    "../models/metadata.json",
                    "../../models/metadata.json",
                    "models/metadata.json"
                };
                bool found = false;
                for (const auto& path : alt_paths) {
                    if (fs::exists(path)) {
                        metadata_path = fs::canonical(path).string();
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("Файл метаданных не найден");
                }
            }
        } catch (const fs::filesystem_error& e) {
            spdlog::error("Ошибка поиска файла метаданных: {}", e.what());
            throw;
        }
        
        spdlog::info("Путь к метаданным: {}", metadata_path);
        load_metadata(metadata_path);
        
        if (fs::exists(preprocessing_path)) {
            spdlog::info("Путь к параметрам предобработки: {}", preprocessing_path);
            load_preprocessing_params(preprocessing_path);
        } else {
            spdlog::warn("Файл параметров предобработки не найден: {}", preprocessing_path);
        }
        
        try {
            if (fs::exists(model_path)) {
                model_path = fs::canonical(model_path).string();
            } else {
                std::vector<std::string> alt_paths = {
                    "../models/rf_nids_csv.onnx",
                    "../../models/rf_nids_csv.onnx",
                    "models/rf_nids_csv.onnx"
                };
                bool found = false;
                for (const auto& path : alt_paths) {
                    if (fs::exists(path)) {
                        model_path = fs::canonical(path).string();
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("ONNX модель не найдена");
                }
            }
        } catch (const fs::filesystem_error& e) {
            spdlog::error("Ошибка поиска ONNX модели: {}", e.what());
            throw;
        }
        
        spdlog::info("Путь к модели: {}", model_path);
        
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        g_session = std::make_unique<Ort::Session>(g_env, model_path.c_str(), opts);
        spdlog::info("ONNX модель успешно загружена");
        
        drogon::app().registerHandler("/predict", &predictHandler, {drogon::Post});
        drogon::app().registerHandler("/health", &healthHandler, {drogon::Get});

        drogon::app().setThreadNum(std::thread::hardware_concurrency());
        drogon::app().setLogLevel(trantor::Logger::kInfo);
        drogon::app().addListener("127.0.0.1", 5001);
        
        spdlog::info("Сервер запущен на http://127.0.0.1:5001");
        spdlog::info("Доступные эндпоинты: POST /predict, GET /health");
        spdlog::info("Числовые признаки: {}", g_preprocessing_params.numerical_features.size());
        spdlog::info("Категориальные признаки: {}", g_preprocessing_params.categorical_features.size());
        
        drogon::app().run();
    } catch (const std::exception& e) {
        spdlog::critical("Критическая ошибка: {}", e.what());
        return 1;
    }
}
