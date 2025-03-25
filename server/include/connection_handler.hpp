#ifndef CONNECTION_HANDLER
#define CONNECTION_HANDlER

#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>

using namespace boost::asio;
using ip::tcp;

namespace Amtal {

class Connection_Handler : public boost::enable_shared_from_this<Connection_Handler>
{

private:
    tcp::socket sock;
    std::string message = "Hello From Server!";
    enum { max_length = 1024 };
    char data[max_length];

public:

    typedef boost::shared_ptr<Connection_Handler> pointer;
    Connection_Handler(boost::asio::io_service& io_service) : sock(io_service) {}

    static pointer create(boost::asio::io_service& io_service) {
        return pointer(new Connection_Handler(io_service));
    }

    tcp::socket& socket() {
        return sock;
    }

    void handle_read(const boost::system::error_code& err, size_t bytes_transferred) {
        if (!err) {
            std::cout << data << "\n";
        }
        else {
            std::cerr << "error: " << err.message() << "\n";
            sock.close();
        }
    }

    void handle_write(const boost::system::error_code& err, size_t bytes_transferred) {
        if (!err) {
            std::cout << "Server sent Hello Message!\n";
        }
        else {
            std::cerr << "error: " << err.message() << "\n";
            sock.close();
        }
    }

    void start() {
        sock.async_read_some(
            boost::asio::buffer(data, max_length),
            boost::bind(&Connection_Handler::handle_read, 
            shared_from_this(),
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
        
        sock.async_write_some(
            boost::asio::buffer(message, max_length),
            boost::bind(&Connection_Handler::handle_write,
                shared_from_this(),
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
 
};

}

#endif