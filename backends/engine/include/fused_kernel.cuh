#pragma once
#include <vector>
#include <cstdint>
#include <span>

struct BatchInput 
{
    std::span<const int32_t> prompt;
    size_t                   cursor;
    void*                    kv_handle;
};

struct BatchOutput 
{
    int32_t next_token;
    bool    finished;
};

class IInferenceEngine 
{
public:
    virtual ~IInferenceEngine() = default;
    virtual void load_model(const std::string& path) = 0;
    virtual void allocate_resources(size_t max_batch, size_t max_seq_len) = 0;
    virtual std::vector<BatchOutput> generate(const std::vector<BatchInput>& batch) = 0;
};
