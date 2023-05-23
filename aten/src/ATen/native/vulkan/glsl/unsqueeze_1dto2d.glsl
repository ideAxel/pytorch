#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Sampler
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uImage;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // info.x: dimension to insert at
  ivec2 info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a new tensor with dimension of size one inserted at the specified
 * position (dim)
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int dim = uBlock.info.x;
  if (dim == 0 || dim == -2) {
    imageStore(uOutput, pos, texelFetch(uImage, pos, 0));
  } else if (dim == 1 || dim == -1) {
    vec4 out_texel = vec4(0, 0, 0, 0);
    int src_x = pos.y;
    int src_y = 0;
    int src_z = 0;
    const vec4 v = texelFetch(uImage, ivec3(src_x, src_y, src_z), 0);
    imageStore(uOutput, pos, v);
  }
}
