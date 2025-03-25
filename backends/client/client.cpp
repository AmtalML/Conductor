#include "include/client.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <boost/asio.hpp>

using namespace boost::asio;
using ip::tcp;

int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cerr << "Usage: client <host> <port>\n";
        return 1;
    }
    try {
        boost::asio::io_service io_service;
        tcp::resolver resolver(io_service);
        tcp::resolver::query query(argv[1], argv[2]);
        tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

        {
            tcp::socket socket(io_service);
            boost::asio::connect(socket, endpoint_iterator);
            std::string http_request =
                "GET / HTTP/1.1\r\nHost: " + std::string(argv[1]) + "\r\n\r\n";
            boost::asio::write(socket, boost::asio::buffer(http_request));

            char reply[1024];
            size_t reply_length = socket.read_some(boost::asio::buffer(reply));
            std::cout << "HTTP Response:\n" << std::string(reply, reply_length) << "\n";
            socket.close();
        }

        {
            tcp::socket socket(io_service);
            boost::asio::connect(socket, endpoint_iterator);
            std::string ai_request =
                "AI_CALL: Analyze this text: 'example'";
            boost::asio::write(socket, boost::asio::buffer(ai_request));

            char reply[1024];
            size_t reply_length = socket.read_some(boost::asio::buffer(reply));
            std::cout << "Completion Response:\n" << std::string(reply, reply_length) << "\n";
            socket.close();
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
