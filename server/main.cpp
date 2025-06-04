// server/main.cpp
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_context.hpp>
#include <thread>
#include <memory>
#include "inference_engine_impl.h"
namespace beast = boost::beast;
namespace http  = beast::http;
namespace asio  = boost::asio;
using tcp       = asio::ip::tcp;
void handle_session(tcp::socket socket, IInferenceEngine* engine) {
    try {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        http::read(socket, buffer, req);
        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::server, "AmtalServer");
        if (req.method() == http::verb::post && req.target() == "/generate") {
            std::vector<BatchInput> batch;
            auto outputs = engine->generate(batch);
            std::string body = R"({"status":"ok","tokens":[]})";
            res.set(http::field::content_type, "application/json");
            res.body() = body;
        } else if (req.method() == http::verb::get && req.target() == "/health") {
            res.set(http::field::content_type, "text/plain");
            res.body() = "alive";
        } else {
            res.result(http::status::not_found);
            res.set(http::field::content_type, "text/plain");
            res.body() = "Not Found";
        }
        res.prepare_payload();
        http::write(socket, res);
    } catch (...) {}
}
int main() {
    const int PORT = 8080;
    std::unique_ptr<IInferenceEngine> engine = std::make_unique<InferenceEngineImpl>();
    engine->load_model("");
    engine->allocate_resources(4, 128);
    try {
        asio::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {tcp::v4(), static_cast<unsigned short>(PORT)}};
        for (;;) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            std::thread{[s = std::move(socket), eng = engine.get()]() mutable {
                handle_session(std::move(s), eng);
            }}.detach();
        }
    } catch (...) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
