#ifndef BATCHING_SERVICE_HPP
#define BATCHING_SERVICE_HPP

#include "utils.hpp"
#include "base_service.hpp"

namespace Amtal {

using Batch = std::vector<Request>;

using ProcessBatchCallBack = std::function<void(const Batch&)>;


class Batch_Scheduler : public Service<int, Request> {

public:

    Batch_Scheduler(size_t max_batch_size, std::chrono::milliseconds flush_time_out, 
    ProcessBatchCallBack call_back);

private:


};


}




#endif 