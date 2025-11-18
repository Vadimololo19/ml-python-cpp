#include <drogon/drogon.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>

using json = nlohmann::json;

std::unique_ptr<Ort::Session> g_session;
Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "ml_service_cpp"};

void predictHandler(const drogon::HttpRequestPtr& req,
                    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    if (req->method() != drogon::Post) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k405MethodNotAllowed);
        callback(resp);
        return;
    }

    try {
        auto j = json::parse(req->body());
        auto features = j.at("features").get<std::vector<float>>();
        if (features.size() != 8) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(R"({"error":"expected 8 features"})");
            callback(resp);
            return;
        }

        std::vector<int64_t> input_shape = {1, 8};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, features.data(), features.size(), input_shape.data(), input_shape.size());

        const char* input_name = "float_input";
        const char* output_name = "variable"; 

        auto output_tensors = g_session->Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1
        );

        float pred = *output_tensors[0].GetTensorData<float>();

        json response{{"prediction", pred}};
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);
    } catch (const std::exception& e) {
        LOG_ERROR << "Handler error: " << e.what();
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setBody(json{{"error", std::string(e.what())}}.dump());
        callback(resp);
    }
}

int main()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  
    g_session = std::make_unique<Ort::Session>(g_env, "model.onnx", opts);

    LOG_INFO << "ONNX model loaded";

    drogon::app().registerHandler("/predict", &predictHandler, {drogon::Post});

    drogon::app().setThreadNum(4);
    drogon::app().setLogLevel(trantor::Logger::kWarn);
    drogon::app().addListener("127.0.0.1", 5001);
    drogon::app().run();
}
