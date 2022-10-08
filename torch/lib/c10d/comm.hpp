#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor to DDP communication hook.
class GradBucket {
 public:
  explicit GradBucket(
      size_t index,
      size_t bucket_count,
      float local_threshold,
      float global_threshold,
      const at::Tensor& tensor,
      const std::vector<size_t>& offsets,
      const std::vector<size_t>& lengths,
      const std::vector<c10::IntArrayRef>& sizes_vec)
      : index_(index),
        bucket_count_(bucket_count),
        local_threshold_(local_threshold),
        global_threshold_(global_threshold),
        tensor_(tensor),
        offsets_(offsets),
        lengths_(lengths),
        sizes_vec_(sizes_vec) {}

  // Returns the index of the bucket, which is unique across all the buckets.
  size_t getIndex() const {
    return index_;
  }

  const at::Tensor& getTensor() const {
    return tensor_;
  }

  // Returns a mutable tensor compared with the above method.
  at::Tensor& getTensorRef() {
    return tensor_;
  }

  // Overwrites tensors at a specific index.
  void setTensor(at::Tensor& tensor) {
    tensor_ = tensor;
  }

  // Each tensor in the list that getPerParameterTensors corresponds to a
  // parameter.
  std::vector<at::Tensor> getPerParameterTensors() const;

  // Returns whether this bucket is the last bucket to allreduce in an iteration.
  bool isTheLastBucketToAllreduceNew(size_t index) const {
    return index == bucket_count_ - 1;
  }

  bool isTheLastBucketToAllreduce() const {
    return index_ == 0;
  }

  size_t getBucketNumber() const {
    return bucket_count_;
  }

  float getLocalThreshold() const {
    return local_threshold_;
  }

  float getGlobalThreshold() const {
    return global_threshold_;
  }

 private:
  size_t index_;
  size_t bucket_count_;
  float local_threshold_;
  float global_threshold_;
  at::Tensor tensor_;
  // Per-variable info in tensors_[0].
  std::vector<size_t> offsets_;
  std::vector<size_t> lengths_;
  std::vector<c10::IntArrayRef> sizes_vec_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing 1) `runHook` method that communicates gradients
// asynchronously, and 2) `parseHookResult` method that converts the hook
// result into a tensor vector.
class TORCH_PYTHON_API CommHookInterface {
 public:
  virtual ~CommHookInterface() {}

  // Passes the input grad bucket to the registered communication hook.
  // Once the tensors in the bucket are ready, kicks off the hook asynchronously
  // and returns a future that holds the communication results.
  virtual c10::intrusive_ptr<c10::ivalue::Future> runHook(
      GradBucket& bucket) = 0;

  // Returns the resulting tensors once the communication hook result is
  // ready. The resulting tensors will then be copied to the grads of
  // individual parameters.
  virtual std::vector<at::Tensor> parseHookResult(
      const c10::IValue& result) = 0;
};

// This CppCommHook interface only requires implementing runHook method that
// potentially uses a state.
// Still need TORCH_PYTHON_API instead of TORCH_API to support Windows platform.
template <typename T>
class TORCH_PYTHON_API CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T& state) : state_(state) {}

  virtual ~CppCommHookInterface() {}

  std::vector<at::Tensor> parseHookResult(const c10::IValue& result) override {
    TORCH_INTERNAL_ASSERT(
        result.isTensor() || result.isTensorList(),
        "expected the hook result is either a Tensor or a TensorList");

    if (result.isTensor()) {
      return {result.toTensor()};
    }

    return result.toTensorVector();
  }

 protected:
  T state_; // Not owned.
};

} // namespace c10d
