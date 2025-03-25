#include "include/utils.hpp"
#include "include/batching_service.hpp"
#include "include/server.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: server <port>\n";
        return 1;
    }
    try {
        boost::asio::io_service io_service;
        Amtal::Server server(io_service, std::atoi(argv[1]));
        io_service.run();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
