// engine/include/inference_engine_impl.h
#pragma once
#include "inference_engine.h"
class InferenceEngineImpl final : public IInferenceEngine {
public:
    InferenceEngineImpl();
    void load_model(const std::string& path) override;
    void allocate_resources(size_t max_batch, size_t max_seq_len) override;
    std::vector<BatchOutput> generate(const std::vector<BatchInput>& batch) override;
};
