#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Block final {
  ivec2 info;
};

Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(
      self.dim() >= 1 || self.dim() <= 3,
      "Vulkan unsqueeze supports 1d, 2d, 3d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() - 1 && dim <= self.dim(),
      "Vulkan unsqueeze dimension out of range expected to be in range of [",
      -self.dim() - 1,
      ",",
      self.dim(),
      "], but got ",
      dim);

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Create the output texture. For unsqueeze, add a dimension.
  std::vector<int64_t> output_size = self.sizes().vec();
  if (dim < 0) {
    dim += (self.dim() + 1);
  }
  output_size.insert(output_size.begin() + dim, 1);
  // Create the output texture
  vTensor v_output{
      context,
      output_size,
      self.scalar_type(),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Total number of work items is equal to the size of the output texture
  uvec3 global_size = v_output.extents();
  // Adaptively determine local work group size, will usually be {4, 4, 4}
  uvec3 local_size = adaptive_work_group_size(global_size);

  // Create the params buffer
  struct Block block {
    {
      // Dimension to unsqueeze
      static_cast<int32_t>(dim),
          // Keep track of the channel in Image3D, for 3d->4d case only.
          static_cast<int32_t>(
              std::ceil(static_cast<float>(output_size[1]) / 4)),
    }
  };
  api::UniformParamsBuffer params(context, block);

  if (self.dim() == 1) {
    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(unsqueeze_1dto2d),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        global_size,
        // local work group size
        local_size,
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
    return convert(v_output);
  }

  else if (self.dim() == 2) {
    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(unsqueeze_2dto3d),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        global_size,
        // local work group size
        local_size,
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
    return convert(v_output);
  }

  else {
    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(unsqueeze_3dto4d),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        global_size,
        // local work group size
        local_size,
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
    return convert(v_output);
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::unsqueeze"), TORCH_FN(unsqueeze));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
