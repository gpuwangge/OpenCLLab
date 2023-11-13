__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

__kernel void read_write_image(read_only image2d_t input_image, write_only image2d_t output_image)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	float tmp = read_imagef(input_image, sampler, (int2) (i,j)).x;
	write_imagef(output_image, (int2) (i,j), (float4) (tmp,0,0,1));
}