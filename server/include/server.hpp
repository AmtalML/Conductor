#ifndef SERVER_HPP
#define SERVER_HPP

#include <iostream>
#include <cstdlib>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include "connection_handler.hpp" 

using namespace boost::asio;
using ip::tcp;

namespace Amtal {

class Server {
public:
    Server(io_service& io_service, unsigned short port)
      : io_service_(io_service), acceptor_(io_service, tcp::endpoint(tcp::v4(), port))
    {
        start_accept();
    }

private:
    void start_accept() {
        Connection_Handler::pointer new_connection =
            Connection_Handler::create(io_service_);
        acceptor_.async_accept(new_connection->socket(),
            boost::bind(&Server::handle_accept, this, new_connection,
              boost::asio::placeholders::error));
    }

    void handle_accept(Connection_Handler::pointer connection,
                       const boost::system::error_code& error) {
        if (!error) {
            connection->start();
        } else {
            std::cerr << "Accept error: " << error.message() << "\n";
        }
        start_accept();
    }

    io_service& io_service_;
    tcp::acceptor acceptor_;
};

}

#endif

