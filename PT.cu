﻿#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define RENDER_WIDTH 1024
#define RENDER_HEIGHT 1024
#define TILE_SIZE 1
#define SPP 1024
#define RR_RATE 0.8
#define PI 3.1415926

// BMP Operation
// 文件信息头结构体
typedef struct
{
    unsigned int   bfSize;        // 文件大小 以字节为单位(2-5字节)
    unsigned short bfReserved1;   // 保留，必须设置为0 (6-7字节)
    unsigned short bfReserved2;   // 保留，必须设置为0 (8-9字节)
    unsigned int   bfOffBits;     // 从文件头到像素数据的偏移  (10-13字节)
} _BITMAPFILEHEADER;

//图像信息头结构体
typedef struct
{
    unsigned int    biSize;          // 此结构体的大小 (14-17字节)
    int             biWidth;         // 图像的宽  (18-21字节)
    int             biHeight;        // 图像的高  (22-25字节)
    unsigned short  biPlanes;        // 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
    unsigned short  biBitCount;      // 一像素所占的位数，一般为24   (28-29字节)
    unsigned int    biCompression;   // 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
    unsigned int    biSizeImage;     // 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
    int             biXPelsPerMeter; // 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
    int             biYPelsPerMeter; // 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
    unsigned int    biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
    unsigned int    biClrImportant;  // 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)
} _BITMAPINFOHEADER;

__host__ void save_image(unsigned char* target_img, int width, int height)
{
    FILE* file_ptr = fopen("RenderResult.bmp", "wb+");

    unsigned short fileType = 0x4d42;
    _BITMAPFILEHEADER fileHeader;
    _BITMAPINFOHEADER infoHeader;

    fileHeader.bfSize = (width) * (height) * 3 + 54;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = (width) * (height) * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    fwrite(&fileType, sizeof(unsigned short), 1, file_ptr);
    fwrite(&fileHeader, sizeof(_BITMAPFILEHEADER), 1, file_ptr);
    fwrite(&infoHeader, sizeof(_BITMAPINFOHEADER), 1, file_ptr);

    fwrite(target_img, sizeof(unsigned char), (height) * (width) * 3, file_ptr);

    fclose(file_ptr);
}

// 3D resources
struct Ray {
    float3 src_point;
    float2 dst_point;
};

struct Trianle {
    float3 tri_a;
    float3 tri_b;
    float3 tri_c;
    float3 normal_line;
    bool is_light;
};
// Light triagles
#define LIGHT_TRI_COUNT 2
/*
Trianle h_light_triangle[] = {
    Trianle{float3{110, 110, 300}, float3{110, 190, 300}, float3{190, 110, 300}, float3{0, 0, -1}},
    Trianle{float3{190, 110, 300}, float3{110, 190, 300}, float3{190, 190, 300}, float3{0, 0, -1}}
};

__constant__ Trianle d_light_triangle[LIGHT_TRI_COUNT];
*/
__constant__ float d_light_irradiance = 40;

// object triagles
// No BVH
#define OBJ_TRI_COUNT 24
Trianle h_scene_objects[] = {
    // light tri
    Trianle{float3{110, 110, 300}, float3{110, 190, 300}, float3{190, 110, 300}, float3{0, 0, -1}, true},
    Trianle{float3{190, 110, 300}, float3{110, 190, 300}, float3{190, 190, 300}, float3{0, 0, -1}, true},
    // internal box 100 * 100 * 30
    // top
    Trianle{float3{100, 100, 100}, float3{200, 100, 100}, float3{100, 200, 100}, float3{0, 0, 1}, false},
    Trianle{float3{200, 100, 100}, float3{200, 200, 100}, float3{100, 200, 100}, float3{0, 0, 1}, false},
    // bottom
    Trianle{float3{100, 100, 70}, float3{200, 100, 70}, float3{100, 200, 70}, float3{0, 0, -1}, false},
    Trianle{float3{200, 100, 70}, float3{200, 200, 70}, float3{100, 200, 70}, float3{0, 0, -1}, false},
    // front
    Trianle{float3{100, 100, 100}, float3{200, 100, 100}, float3{100, 100, 70}, float3{0, -1, 0}, false},
    Trianle{float3{100, 100, 70}, float3{200, 100, 70}, float3{200, 100, 100}, float3{0, -1, 0}, false},

    // behind
    Trianle{float3{100, 200, 100}, float3{200, 200, 100}, float3{100, 200, 70}, float3{0, 1, 0}, false},
    Trianle{float3{100, 200, 70}, float3{200, 200, 70}, float3{200, 200, 100}, float3{0, 1, 0}, false},

    // left
    Trianle{float3{100, 100, 100}, float3{100, 200, 100}, float3{100, 100, 70}, float3{-1, 0, 0}, false},
    Trianle{float3{100, 100, 70}, float3{100, 200, 70}, float3{100, 200, 100}, float3{-1, 0, 0}, false},

    // right
    Trianle{float3{200, 100, 100}, float3{200, 200, 100}, float3{200, 100, 70}, float3{1, 0, 0}, false},
    Trianle{float3{200, 100, 70}, float3{200, 200, 70}, float3{200, 200, 100}, float3{1, 0, 0}, false},

    // general box 300 * 300 * 310
    // top
    Trianle{float3{0, 0, 300}, float3{0, 300, 300}, float3{300, 0, 300}, float3{0, 0, -1}, false},
    Trianle{float3{0, 300, 300}, float3{300, 0, 300}, float3{300, 300, 300}, float3{0, 0, -1}, false},

    // bottom
    Trianle{float3{0, 0, 0}, float3{0, 300, 0}, float3{300, 0, 0}, float3{0, 0, 1}, false},
    Trianle{float3{0, 300, 0}, float3{300, 0, 0}, float3{300, 300, 0}, float3{0, 0, 1}, false},

    // left
    Trianle{float3{0, 0, 0}, float3{0, 0, 300}, float3{0, 300, 300}, float3{1, 0, 0}, false},
    Trianle{float3{0, 300, 300}, float3{0, 300, 0}, float3{0, 0, 0}, float3{1, 0, 0}, false},

    // right
    Trianle{float3{300, 0, 0}, float3{300, 0, 300}, float3{300, 300, 300}, float3{-1, 0, 0}, false},
    Trianle{float3{300, 300, 300}, float3{300, 300, 0}, float3{300, 0, 0}, float3{-1, 0, 0}, false},

    // behind
    Trianle{float3{0, 300, 0}, float3{0, 300, 300}, float3{300, 300, 0}, float3{0, -1, 0}, false},
    Trianle{float3{300, 300, 0}, float3{300, 300, 300}, float3{0, 300, 300}, float3{0, -1, 0}, false}
};

__constant__ Trianle d_scene_objects[OBJ_TRI_COUNT];

#define BRDF_rate 0.5

// camera position
__constant__ float3 d_camera_position = float3{150, -400, 150};
__constant__ float3 d_camera_direction = float3{0, 1, 0};
// 浮点精度考虑，设置较大焦距和成像平面
__constant__ float d_camera_focal_length = 200;
__constant__ float d_camera_width = 150;
__constant__ float d_camera_height = 150;
__constant__ float d_camera_pixel_width = 150.0 / RENDER_WIDTH;
__constant__ float d_camera_pixel_height= 150.0 / RENDER_HEIGHT;


__device__ inline float mixed_product(float3 vec_a, float3 vec_b, float3 vec_c)
{
    return vec_a.x * (vec_b.y * vec_c.z - vec_b.z * vec_c.y) + 
        vec_a.y * (vec_b.z * vec_c.x - vec_b.x * vec_c.z) + 
        vec_a.z * (vec_b.x * vec_c.y - vec_b.y * vec_c.x);
}


__device__ inline float3 sub_float3(float3 opr1, float3 opr2)
{
    return make_float3(opr1.x - opr2.x, opr1.y - opr2.y, opr1.z - opr2.z);
}


__device__ inline float3 scalar_mult_float3(float3 vec, float scalar)
{
    return make_float3(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

__device__ float dot(float3 opr1, float3 opr2)
{
    return opr1.x * opr2.x + opr1.y * opr2.y + opr1.z * opr2.z;
}

__device__ inline float3 add_float3(float3 opr1, float3 opr2)
{
    return make_float3(opr1.x + opr2.x, opr1.y + opr2.y, opr1.z + opr2.z);
}


__device__ float size(Trianle triangle)
{
    float3 vec1 = sub_float3(triangle.tri_b, triangle.tri_a);
    float3 vec2 = sub_float3(triangle.tri_c, triangle.tri_a);
    float3 cross_product = make_float3(vec1.y * vec2.z - vec1.z * vec2.y, vec1.z * vec2.x - vec1.x * vec2.z, vec1.x * vec2.y - vec1.y * vec2.x);
    return 0.5 * norm3df(cross_product.x, cross_product.y, cross_product.z);
}


__device__ float3 check_obj_hit(int src_tri_idx, float3 src_point, float3 direction, int& hit_obj_idx)
{
    // normalize direction
    float div_length = 1 / norm3df(direction.x, direction.y, direction.z);
    float3 normal_direction = make_float3(direction.x * div_length, direction.y * div_length, direction.z * div_length);

    hit_obj_idx = -1;

    float3 hit_point;
    float min_distance = 2147483647;

    for (int i = 0; i < OBJ_TRI_COUNT; ++i) {
        if (i == src_tri_idx) {
            continue;
        }
        // make shadow
        Trianle shadow_tri = Trianle{sub_float3(d_scene_objects[i].tri_a, scalar_mult_float3(normal_direction, dot(normal_direction, sub_float3(d_scene_objects[i].tri_a, src_point)))),
            sub_float3(d_scene_objects[i].tri_b, scalar_mult_float3(normal_direction, dot(normal_direction, sub_float3(d_scene_objects[i].tri_b, src_point)))),
            sub_float3(d_scene_objects[i].tri_c, scalar_mult_float3(normal_direction, dot(normal_direction, sub_float3(d_scene_objects[i].tri_c, src_point)))),
            normal_direction};

        // check in center
        float3 vec_pa = sub_float3(shadow_tri.tri_a, src_point);
        float3 vec_pb = sub_float3(shadow_tri.tri_b, src_point);
        float3 vec_pc = sub_float3(shadow_tri.tri_c, src_point);

        float papb = mixed_product(normal_direction, vec_pa, vec_pb);
        float pbpc = mixed_product(normal_direction, vec_pb, vec_pc);
        float pcpa = mixed_product(normal_direction, vec_pc, vec_pa);
        if ((papb > 0 && pbpc > 0 && pcpa > 0) || (papb < 0 && pbpc < 0 && pcpa < 0)) {
            // in center
            // get hit point
            // get coordinary, reuse vec_pb ,vec_pc
            vec_pb = sub_float3(shadow_tri.tri_b, shadow_tri.tri_a);
            vec_pc = sub_float3(shadow_tri.tri_c, shadow_tri.tri_a);
            vec_pa = sub_float3(src_point, shadow_tri.tri_a);
            float divider = vec_pb.x * vec_pc.y - vec_pb.y * vec_pc.x;
            float rate_a = (vec_pc.y * vec_pa.x - vec_pc.x * vec_pa.y) / divider;
            float rate_b = (-vec_pb.y * vec_pa.x + vec_pb.x * vec_pa.y) / divider;

            vec_pb = sub_float3(d_scene_objects[i].tri_b, d_scene_objects[i].tri_a);
            vec_pc = sub_float3(d_scene_objects[i].tri_c, d_scene_objects[i].tri_a);
            vec_pa.x = d_scene_objects[i].tri_a.x + rate_a * vec_pb.x + rate_b * vec_pc.x;
            vec_pa.y = d_scene_objects[i].tri_a.y + rate_a * vec_pb.y + rate_b * vec_pc.y;
            vec_pa.z = d_scene_objects[i].tri_a.z + rate_a * vec_pb.z + rate_b * vec_pc.z;

            float distance = dot(sub_float3(vec_pa, src_point), normal_direction);
            // printf("Rate : %f %f %f\n", rate_a, rate_b, distance / norm3df(vec_pa.x - src_point.x, vec_pa.y - src_point.y, vec_pa.z - src_point.z));
            if (distance > 0) {
                // printf("In Center : %f, %f, %f %f\n", papb, pbpc, pcpa, distance);
                // ray will hit object
                if (distance < min_distance) {
                    min_distance = distance;
                    hit_point = vec_pa;
                    hit_obj_idx = i;
                }
            }
        }
    }

    // printf("Src : %d   Dst : %d   Direction : %f, %f, %f\n", src_tri_idx, hit_obj_idx, direction.x, direction.y, direction.z);
    return hit_point;
}



__device__ float3 check_light_hit(int src_tri_idx, float3 src_point, float3 direction, int& hit_obj_idx)
{
    float3 hit_point = check_obj_hit(src_tri_idx, src_point, direction, hit_obj_idx);
    if (hit_obj_idx > -1 && !d_scene_objects[hit_obj_idx].is_light) {
        hit_obj_idx = -1;
    }

    return hit_point;
}


__device__ float shade(int object_idx, float3 src_point, float3 direction, curandState* curand_state)
{
    // Contribution from the light source.
    float l_dir = 0;
    for (int i = 0; i < LIGHT_TRI_COUNT; ++i) {
        // random select a point on light triangle
        float rand_x = curand_uniform(curand_state);
        float rand_y = curand_uniform(curand_state);
        if (rand_x + rand_y > 1) {
            rand_x = 1 - rand_x;
            rand_y = 1 - rand_y;
        }
        float3 random_point = add_float3(d_scene_objects[i].tri_a, add_float3(scalar_mult_float3(sub_float3(d_scene_objects[i].tri_b, d_scene_objects[i].tri_a), rand_x), scalar_mult_float3(sub_float3(d_scene_objects[i].tri_c, d_scene_objects[i].tri_a), rand_y)));

        // test block
        float3 obj_light_direction = sub_float3(random_point, src_point);
        int test_block_idx;
        check_obj_hit(-1, src_point, obj_light_direction, test_block_idx);
        // printf("Direction %f %f %f %d\n", obj_light_direction.x, obj_light_direction.y, obj_light_direction.z, test_block_idx);
        if (test_block_idx == i) {
            // printf("Hit Light!\n");
            float direction_length_square = obj_light_direction.x * obj_light_direction.x + obj_light_direction.y * obj_light_direction.y + obj_light_direction.z * obj_light_direction.z;
            l_dir += d_light_irradiance * BRDF_rate * dot(d_scene_objects[object_idx].normal_line, obj_light_direction) * -1 * dot(d_scene_objects[i].normal_line, obj_light_direction) 
                        / direction_length_square / direction_length_square * size(d_scene_objects[i]);
            // printf("Shade %d %f %f\n", i, dot(d_light_triangle[i].normal_line, obj_light_direction), l_dir);
        }
    }

    // Contribution from other reflectors.
    float l_indir = 0;

    // test Russian Roulette
    float rr_result = curand_uniform(curand_state);
    if (rr_result < RR_RATE) {
        // random select a ray from src_point
        float cosine_theta = 2 * (curand_uniform(curand_state) - 0.5);
        float sine_theta = sqrtf(1 - cosine_theta * cosine_theta);
        float fai_value = 2 * PI * curand_uniform(curand_state);
        float3 ray_direction = make_float3(sine_theta * cosf(fai_value), sine_theta * sinf(fai_value), cosine_theta);
        if (dot(ray_direction, d_scene_objects[object_idx].normal_line) < 0) {
            ray_direction.x *= -1;
            ray_direction.y *= -1;
            ray_direction.z *= -1;
            cosine_theta *= -1;
        }

        int hit_obj_idx;
        float3 hit_point = check_obj_hit(object_idx, src_point, ray_direction, hit_obj_idx);
        if (hit_obj_idx > -1) {
            // printf("Hit Object!\n");
            ray_direction.x *= -1;
            ray_direction.y *= -1;
            ray_direction.z *= -1;
            l_indir = shade(hit_obj_idx, hit_point, ray_direction, curand_state) * BRDF_rate * dot(ray_direction, d_scene_objects[hit_obj_idx].normal_line) * 2 * PI / RR_RATE;
        }
    }

    // printf("Shade %f\n", l_dir + l_indir);
    return l_dir + l_indir;
}


__device__ float ray_generation(float3 pixel_center_position, curandState* curand_states)
{
    float pixel_radiance = 0;
    for (int i = 0; i < SPP; ++i) {
        float width_bias = d_camera_pixel_width * (curand_uniform(&curand_states[threadIdx.x]) - 0.5);
        float height_bias = d_camera_pixel_height * (curand_uniform(&curand_states[threadIdx.x]) - 0.5);
        int hit_obj_idx;
        // printf("Pixel bias : %f %f\n", width_bias, height_bias);


        float3 ray_direction = sub_float3(add_float3(pixel_center_position, make_float3(width_bias, 0, height_bias)), d_camera_position);
        float3 hit_light_point = check_light_hit(-1, d_camera_position, ray_direction, hit_obj_idx);
        if (hit_obj_idx > -1) {
            // printf("Ray Hit!\n");
            pixel_radiance += 1.0 / SPP * d_light_irradiance;
        }
        /*
        else {
            float3 hit_point = check_obj_hit(-1, d_camera_position, ray_direction, hit_obj_idx);
            if (hit_obj_idx > -1) {
                // printf("Obj Hit!\n");
                float3 reverse_ray_direction = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
                pixel_radiance += 1.0 / SPP * shade(hit_obj_idx, hit_point, reverse_ray_direction, &curand_states[threadIdx.x]);
                // printf("Ray Obj General : %f\n", pixel_radiance);
            }
        }
        */
    }

    // printf("Ray General : %f\n", pixel_radiance);
    return pixel_radiance;
}



__global__ void render_pixel(unsigned char* target_img, curandState* curand_states)
{
    int target_pixel_width = blockIdx.x * TILE_SIZE + threadIdx.x;
    int target_pixel_height = blockIdx.y * TILE_SIZE + threadIdx.y;
    // printf("%d, %d\n", target_pixel_width, target_pixel_height);

    float3 delta = make_float3((target_pixel_width + 0.5 - RENDER_WIDTH / 2.0) * d_camera_pixel_width, d_camera_focal_length, (target_pixel_height + 0.5 - RENDER_HEIGHT / 2.0) * d_camera_pixel_height);
    float3 pixel_center = make_float3(d_camera_position.x + delta.x, d_camera_position.y + delta.y, d_camera_position.z + delta.z);
    float pixel_radiance = ray_generation(pixel_center, curand_states);
    // float pixel_radiance = 20;

    // printf("%d, %d : %f\n", target_pixel_width, target_pixel_height, pixel_radiance);
    if (pixel_radiance > d_light_irradiance) {
        pixel_radiance = d_light_irradiance;
    }
    unsigned char rgb_value = (unsigned char)(pixel_radiance / d_light_irradiance * 255);
    // printf("%d, %d : %d\n", target_pixel_width, target_pixel_height, rgb_value);
    int base_idx = 3 * (target_pixel_height * RENDER_WIDTH + target_pixel_width);
    target_img[base_idx] = 55;
    target_img[base_idx + 1] = rgb_value;
    target_img[base_idx + 2] = rgb_value;
}


__global__ void init_curand(curandState* curand_states, int seed)
{
    curand_init(seed, threadIdx.x, 0, &(curand_states[threadIdx.x]));
}

int main()
{
    dim3 grid{RENDER_WIDTH / TILE_SIZE, RENDER_HEIGHT / TILE_SIZE, 1};
    dim3 block{TILE_SIZE, TILE_SIZE, 1};

    unsigned char* d_target_img;
    cudaMalloc(&d_target_img, RENDER_WIDTH * RENDER_HEIGHT * 3);

    curandState* curand_states;
    cudaMalloc(&curand_states, TILE_SIZE * sizeof(curandState));

    init_curand <<<1, TILE_SIZE>>> (curand_states, 0);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "curand init launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaMemcpyToSymbol(d_light_triangle, h_light_triangle, sizeof(Trianle) * LIGHT_TRI_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_scene_objects, h_scene_objects, sizeof(Trianle) * OBJ_TRI_COUNT, cudaMemcpyHostToDevice);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "before render launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    render_pixel <<<grid, block>>> (d_target_img, curand_states);
    
    unsigned char* h_target_img = (unsigned char*)malloc(RENDER_WIDTH * RENDER_HEIGHT * 3);

    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "render launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaMemcpy(h_target_img, d_target_img, RENDER_WIDTH * RENDER_HEIGHT * 3, cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "copy launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    save_image(h_target_img, RENDER_WIDTH, RENDER_HEIGHT);
    free(h_target_img);

    cudaFree(d_target_img);
    cudaFree(curand_states);
    cudaDeviceReset();

    return 0;
}