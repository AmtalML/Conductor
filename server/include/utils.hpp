#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace Amtal {

struct Request {
    int id;
    std::string prompt;
};

}



#endif 