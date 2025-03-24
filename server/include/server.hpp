#ifndef SERVER_HPP
#define SERVER_HPP

#include <boost/asio.hpp>

namespace beast = boost::beast;
namespace http = beast:: http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace Amtal {

class Server {

    net::io_context& ioc_;
    tcp::acceptor acceptor_;

public:



private:




};

}

#endif 