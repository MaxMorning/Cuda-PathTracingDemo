#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define RENDER_WIDTH 512
#define RENDER_HEIGHT 512
#define TILE_SIZE 16
#define STACK_CAPACITY 128
#define SHARED_MEM_CAP STACK_CAPACITY * RENDER_WIDTH * RENDER_HEIGHT
#define SPP 1024
#define RR_RATE 0.9
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

struct Trianle {
    float3 tri_a;
    float3 tri_b;
    float3 tri_c;
    float3 normal_line;
    bool is_light;
    float brdf_rate;
};

/*
// test scene
// Light triagles
#define LIGHT_TRI_COUNT 2

__constant__ float d_light_irradiance = 40;

// object triagles
// No BVH
#define BRDF_rate 0.6

#define OBJ_TRI_COUNT 24
Trianle h_scene_objects[] = {
    // light tri
    Trianle{float3{110, 110, 300}, float3{110, 190, 300}, float3{190, 110, 300}, float3{0, 0, -1}, true, BRDF_rate},
    Trianle{float3{190, 110, 300}, float3{110, 190, 300}, float3{190, 190, 300}, float3{0, 0, -1}, true, BRDF_rate},
    // Trianle{float3{110, 110, 301}, float3{110, 190, 301}, float3{190, 110, 301}, float3{0, 0, 1}, true, BRDF_rate},
    // Trianle{float3{190, 110, 301}, float3{110, 190, 301}, float3{190, 190, 301}, float3{0, 0, 1}, true, BRDF_rate},
    // internal box 100 * 100 * 30
    // top
    Trianle{float3{100, 100, 100}, float3{200, 100, 100}, float3{100, 200, 100}, float3{0, 0, 1}, false, BRDF_rate},
    Trianle{float3{200, 100, 100}, float3{200, 200, 100}, float3{100, 200, 100}, float3{0, 0, 1}, false, BRDF_rate},
    // bottom
    Trianle{float3{100, 100, 70}, float3{200, 100, 70}, float3{100, 200, 70}, float3{0, 0, -1}, false, BRDF_rate},
    Trianle{float3{200, 100, 70}, float3{200, 200, 70}, float3{100, 200, 70}, float3{0, 0, -1}, false, BRDF_rate},
    // front
    Trianle{float3{100, 100, 100}, float3{200, 100, 100}, float3{100, 100, 70}, float3{0, -1, 0}, false, BRDF_rate},
    Trianle{float3{100, 100, 70}, float3{200, 100, 70}, float3{200, 100, 100}, float3{0, -1, 0}, false, BRDF_rate},

    // behind
    Trianle{float3{100, 200, 100}, float3{200, 200, 100}, float3{100, 200, 70}, float3{0, 1, 0}, false, BRDF_rate},
    Trianle{float3{100, 200, 70}, float3{200, 200, 70}, float3{200, 200, 100}, float3{0, 1, 0}, false, BRDF_rate},

    // left
    Trianle{float3{100, 100, 100}, float3{100, 200, 100}, float3{100, 100, 70}, float3{-1, 0, 0}, false, BRDF_rate},
    Trianle{float3{100, 100, 70}, float3{100, 200, 70}, float3{100, 200, 100}, float3{-1, 0, 0}, false, BRDF_rate},

    // right
    Trianle{float3{200, 100, 100}, float3{200, 200, 100}, float3{200, 100, 70}, float3{1, 0, 0}, false, BRDF_rate},
    Trianle{float3{200, 100, 70}, float3{200, 200, 70}, float3{200, 200, 100}, float3{1, 0, 0}, false, BRDF_rate},

    // general box 300 * 300 * 300.001
    // top
    Trianle{float3{0, 0, 300.001}, float3{0, 300, 300.001}, float3{300, 0, 300.001}, float3{0, 0, -1}, false, BRDF_rate},
    Trianle{float3{0, 300, 300.001}, float3{300, 0, 300.001}, float3{300, 300, 300.001}, float3{0, 0, -1}, false, BRDF_rate},

    // bottom
    Trianle{float3{0, 0, 0}, float3{0, 300, 0}, float3{300, 0, 0}, float3{0, 0, 1}, false, BRDF_rate},
    Trianle{float3{0, 300, 0}, float3{300, 0, 0}, float3{300, 300, 0}, float3{0, 0, 1}, false, BRDF_rate},

    // left
    Trianle{float3{0, 0, 0}, float3{0, 0, 300.001}, float3{0, 300, 300.001}, float3{1, 0, 0}, false, BRDF_rate},
    Trianle{float3{0, 300, 300.001}, float3{0, 300, 0}, float3{0, 0, 0}, float3{1, 0, 0}, false, BRDF_rate},

    // right
    Trianle{float3{300, 0, 0}, float3{300, 0, 300.001}, float3{300, 300, 300.001}, float3{-1, 0, 0}, false, BRDF_rate},
    Trianle{float3{300, 300, 300.001}, float3{300, 300, 0}, float3{300, 0, 0}, float3{-1, 0, 0}, false, BRDF_rate},

    // behind
    Trianle{float3{0, 300, 0}, float3{0, 300, 300.001}, float3{300, 300, 0}, float3{0, -1, 0}, false, BRDF_rate},
    Trianle{float3{300, 300, 0}, float3{300, 300, 300.001}, float3{0, 300, 300.001}, float3{0, -1, 0}, false, BRDF_rate}
};

__constant__ Trianle d_scene_objects[OBJ_TRI_COUNT];


// camera position
__constant__ float3 d_camera_position = float3{150, -400, 150};
__constant__ float3 d_camera_direction = float3{0, 1, 0};
__constant__ float3 d_camera_up_direction = float3{0, 0, 1};
__constant__ float3 d_camera_left_direction = float3{1, 0, 0};
// 浮点精度考虑，设置较大焦距和成像平面
__constant__ float d_camera_focal_length = 200;
__constant__ float d_camera_width = 150;
__constant__ float d_camera_height = 150;
__constant__ float d_camera_pixel_width = 150.0 / RENDER_WIDTH;
__constant__ float d_camera_pixel_height= 150.0 / RENDER_HEIGHT;
*/



// Cornell box
#define LIGHT_TRI_COUNT 2
__constant__ float d_light_irradiance = 42;

#define BRDF_rate 0.74
#define OBJ_TRI_COUNT 32
// Trianle{float3{}, float3{}, float3{}, float3{}, false, BRDF_rate},
Trianle h_scene_objects[] = {
    // Light triagles
    Trianle{float3{343.0, 548.799, 227.0}, float3{343.0, 548.799, 332.0}, float3{213.0, 548.799, 332.0}, float3{0, -1, 0}, true, BRDF_rate},
    Trianle{float3{343.0, 548.799, 227.0}, float3{213.0, 548.799, 227.0}, float3{213.0, 548.799, 332.0}, float3{0, -1, 0}, true, BRDF_rate},
    
    // Floor
    Trianle{float3{552.8, 0.0, 0.0}, float3{0.0, 0.0, 0.0}, float3{0.0, 0.0, 559.2}, float3{0, 1, 0}, false, BRDF_rate},
    Trianle{float3{552.8, 0.0, 0.0}, float3{549.6, 0.0, 559.2}, float3{0.0, 0.0, 559.2}, float3{0, 1, 0}, false, BRDF_rate},

    // Ceiling
    Trianle{float3{556.0, 548.8, 0.0}, float3{556.0, 548.8, 559.2}, float3{0.0, 548.8, 559.2}, float3{0, -1, 0}, false, BRDF_rate},
    Trianle{float3{556.0, 548.8, 0.0}, float3{0.0, 548.8, 0.0}, float3{0.0, 548.8, 559.2}, float3{0, -1, 0}, false, BRDF_rate},

    // Back wall
    Trianle{float3{549.6, 0.0, 559.2}, float3{0.0, 0.0, 559.2}, float3{0.0, 548.8, 559.2}, float3{0, 0, -1}, false, BRDF_rate},
    Trianle{float3{549.6, 0.0, 559.2}, float3{556.0, 548.8, 559.2}, float3{0.0, 548.8, 559.2}, float3{0, 0, -1}, false, BRDF_rate},

    // Right wall
    Trianle{float3{0.0, 0.0, 559.2}, float3{0.0, 0.0, 0.0}, float3{0.0, 548.8, 0.0}, float3{1, 0, 0}, false, BRDF_rate},
    Trianle{float3{0.0, 0.0, 559.2}, float3{0.0, 548.8, 559.2}, float3{0.0, 548.8, 0.0}, float3{1, 0, 0}, false, BRDF_rate},

    // Left wall
    Trianle{float3{552.8, 0.0, 0.0}, float3{549.6, 0.0, 559.2}, float3{556.0, 548.8, 559.2}, float3{-1, 0, 0}, false, BRDF_rate},
    Trianle{float3{552.8, 0.0, 0.0}, float3{556.0, 548.8, 0.0}, float3{556.0, 548.8, 559.2}, float3{-1, 0, 0}, false, BRDF_rate},
    
    // Short block
    // Top
    Trianle{float3{130.0, 165.0, 65.0}, float3{82.0, 165.0, 225.0}, float3{240.0, 165.0, 272.0}, float3{0, 1, 0}, false, BRDF_rate},
    Trianle{float3{130.0, 165.0, 65.0}, float3{290.0, 165.0, 114.0}, float3{240.0, 165.0, 272.0}, float3{0, 1, 0}, false, BRDF_rate},
    
    // Left
    Trianle{float3{290.0, 0.0, 114.0}, float3{290.0, 165.0, 114.0}, float3{240.0, 165.0, 272.0}, float3{-0.9534, 0, -0.301709}, false, BRDF_rate},
    Trianle{float3{290.0, 0.0, 114.0}, float3{240.0, 0.0, 272.0}, float3{240.0, 165.0, 272.0}, float3{-0.9534, 0, -0.301709}, false, BRDF_rate},
    
    // Front
    Trianle{float3{130.0, 0.0, 65.0}, float3{130.0, 165.0, 65.0}, float3{290.0, 165.0, 114.0}, float3{-0.292826, 0, -0.956166}, false, BRDF_rate},
    Trianle{float3{130.0, 0.0, 65.0}, float3{290.0, 0.0, 114.0}, float3{290.0, 165.0, 114.0}, float3{-0.292826, 0, -0.956166}, false, BRDF_rate},

    // Right
    Trianle{float3{82.0, 0.0, 225.0}, float3{82.0, 165.0, 225.0}, float3{130.0, 165.0, 65.0}, float3{-0.957826, 0, -0.287348}, false, BRDF_rate},
    Trianle{float3{82.0, 0.0, 225.0}, float3{130.0, 0.0, 65.0}, float3{130.0, 165.0, 65.0}, float3{-0.957826, 0, -0.287348}, false, BRDF_rate},
    
    // Behind
    Trianle{float3{240.0, 0.0, 272.0}, float3{240.0, 165.0, 272.0}, float3{82.0, 165.0, 225.0}, float3{-0.285121, 0, -0.958492}, false, BRDF_rate},
    Trianle{float3{240.0, 0.0, 272.0}, float3{82.0, 0.0, 225.0}, float3{82.0, 165.0, 225.0}, float3{-0.285121, 0, -0.958492}, false, BRDF_rate},

    // Tall block
    // Top
    Trianle{float3{423.0, 330.0, 247.0}, float3{265.0, 330.0, 296.0}, float3{314.0, 330.0, 456.0}, float3{0, 1, 0}, false, BRDF_rate},
    Trianle{float3{423.0, 330.0, 247.0}, float3{472.0, 330.0, 406.0}, float3{314.0, 330.0, 456.0}, float3{0, 1, 0}, false, BRDF_rate},

    // Left
    Trianle{float3{423.0, 0.0, 247.0}, float3{423.0, 330.0, 247.0}, float3{472.0, 330.0, 406.0}, float3{0.955649, 0, -0.294508}, false, BRDF_rate},
    Trianle{float3{423.0, 0.0, 247.0}, float3{472.0, 0.0, 406.0}, float3{472.0, 330.0, 406.0}, float3{0.955649, 0, -0.294508}, false, BRDF_rate},

    // Behind
    Trianle{float3{472.0, 0.0, 406.0}, float3{472.0, 330.0, 406.0}, float3{314.0, 330.0, 456.0}, float3{-0.301709, 0, -0.953400}, false, BRDF_rate},
    Trianle{float3{472.0, 0.0, 406.0}, float3{314.0, 0.0, 456.0}, float3{314.0, 330.0, 456.0}, float3{-0.301709, 0, -0.953400}, false, BRDF_rate},

    // Right
    Trianle{float3{314.0, 0.0, 456.0}, float3{314.0, 330.0, 456.0}, float3{265.0, 330.0, 296.0}, float3{0.956166, 0, -0.292826}, false, BRDF_rate},
    Trianle{float3{314.0, 0.0, 456.0}, float3{265.0, 0.0, 296.0}, float3{265.0, 330.0, 296.0}, float3{0.956166, 0, -0.292826}, false, BRDF_rate},
    
    // Front
    Trianle{float3{265.0, 0.0, 296.0}, float3{265.0, 330.0, 296.0}, float3{423.0, 330.0, 247.0}, float3{-0.296209, 0, -0.955123}, false, BRDF_rate},
    Trianle{float3{265.0, 0.0, 296.0}, float3{423.0, 0.0, 247.0}, float3{423.0, 330.0, 247.0}, float3{-0.296209, 0, -0.955123}, false, BRDF_rate}
};

__constant__ Trianle d_scene_objects[OBJ_TRI_COUNT];
// camera position
__constant__ float3 d_camera_position = float3{278, 273, -800};
__constant__ float3 d_camera_direction = float3{0, 0, 1};
__constant__ float3 d_camera_up_direction = float3{0, 1, 0};
__constant__ float3 d_camera_left_direction = float3{-1, 0, 0};

__constant__ float d_camera_focal_length = 3.5;
__constant__ float d_camera_width = 2.5;
__constant__ float d_camera_height = 2.5;
__constant__ float d_camera_pixel_width = 2.5 / RENDER_WIDTH;
__constant__ float d_camera_pixel_height= 2.5 / RENDER_HEIGHT;


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

/*
__device__ float shade_recurse(int object_idx, float3 src_point, float3 direction, curandState* curand_state)
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

    return l_dir;
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
        if (hit_obj_idx > -1 && !d_scene_objects[hit_obj_idx].is_light) {
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
*/
__device__ float stack_dir[SHARED_MEM_CAP];
__device__ float stack_indir_rate[SHARED_MEM_CAP];

__device__ float shade(int object_idx, float3 src_point, float3 direction, curandState* curand_state)
{
    // __shared__ float stack_dir[SHARED_MEM_CAP];
    // __shared__ float stack_indir_rate[SHARED_MEM_CAP];

    // int stack_size = 0;
    float l_dir = 0;
    int stack_offset = ((blockIdx.y * TILE_SIZE + threadIdx.y) * RENDER_WIDTH + (blockIdx.x * TILE_SIZE + threadIdx.x)) * STACK_CAPACITY;
    int stack_ori = stack_offset;
    float3 out_direction = direction; // use in BRDF, here is ignored.
    float3 ray_src = src_point;
    int src_object_idx = object_idx;
    while (true) {
        // Contribution from the light source.
        l_dir = 0;
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
            float3 obj_light_direction = sub_float3(random_point, ray_src);
            int test_block_idx;
            check_obj_hit(-1, ray_src, obj_light_direction, test_block_idx);
            // printf("Direction %f %f %f %d\n", obj_light_direction.x, obj_light_direction.y, obj_light_direction.z, test_block_idx);
            if (test_block_idx == i) {
                // printf("Hit Light!\n");
                float direction_length_square = obj_light_direction.x * obj_light_direction.x + obj_light_direction.y * obj_light_direction.y + obj_light_direction.z * obj_light_direction.z;
                l_dir += d_light_irradiance * d_scene_objects[src_object_idx].brdf_rate * dot(d_scene_objects[src_object_idx].normal_line, obj_light_direction) * -1 * dot(d_scene_objects[i].normal_line, obj_light_direction) 
                            / direction_length_square / direction_length_square * size(d_scene_objects[i]);
                // printf("Shade %d %f %f\n", i, dot(d_light_triangle[i].normal_line, obj_light_direction), l_dir);
            }
        }

        // Contribution from other reflectors.
        // test Russian Roulette
        float rr_result = curand_uniform(curand_state);
        if (rr_result < RR_RATE) {
            float indir_rate = 0;
            // random select a ray from src_point
            float cosine_theta = 2 * (curand_uniform(curand_state) - 0.5);
            float sine_theta = sqrtf(1 - cosine_theta * cosine_theta);
            float fai_value = 2 * PI * curand_uniform(curand_state);
            float3 ray_direction = make_float3(sine_theta * cosf(fai_value), sine_theta * sinf(fai_value), cosine_theta);
            if (dot(ray_direction, d_scene_objects[src_object_idx].normal_line) < 0) {
                ray_direction.x *= -1;
                ray_direction.y *= -1;
                ray_direction.z *= -1;
                cosine_theta *= -1;
            }

            int hit_obj_idx;
            float3 hit_point = check_obj_hit(src_object_idx, ray_src, ray_direction, hit_obj_idx);
            if (hit_obj_idx > -1 && !d_scene_objects[hit_obj_idx].is_light) {
                // printf("Hit Object!\n");
                ray_direction.x *= -1;
                ray_direction.y *= -1;
                ray_direction.z *= -1;
                indir_rate = d_scene_objects[hit_obj_idx].brdf_rate * dot(ray_direction, d_scene_objects[hit_obj_idx].normal_line) / RR_RATE;
                src_object_idx = hit_obj_idx;
                ray_src = hit_point;
                out_direction = ray_direction;

                stack_dir[stack_offset] = l_dir;
                stack_indir_rate[stack_offset] = indir_rate;
                ++stack_offset;
            }
            else {
                // stack_dir[stack_offset] = l_dir;
                // stack_indir_rate[stack_offset] = indir_rate;
                // ++stack_offset;
                break;
            }
        }
        else {
            break;
        }
    }

    // calc final irradiance
    for (int i = stack_offset - 1; i >= stack_ori; --i) {
        // printf("%f %f\n", stack_indir_rate[i], stack_dir[i]);
        l_dir *= stack_indir_rate[i];
        l_dir += stack_dir[i];
    }
    return l_dir;
}

__device__ __forceinline__ float ray_generation(float3 pixel_center_position, curandState* curand_states)
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
        else {
            float3 hit_point = check_obj_hit(-1, d_camera_position, ray_direction, hit_obj_idx);
            if (hit_obj_idx > -1) {
                // printf("Obj Hit!\n");
                float3 reverse_ray_direction = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
                pixel_radiance += 1.0 / SPP * shade(hit_obj_idx, hit_point, reverse_ray_direction, &curand_states[threadIdx.x]);
                // printf("Ray Obj General : %f\n", pixel_radiance);
            }
        }
    }

    // printf("Ray General : %f\n", 1.0 / SPP * d_light_irradiance);
    return pixel_radiance;
}



__global__ void render_pixel(unsigned char* target_img, curandState* curand_states)
{
    int target_pixel_width = blockIdx.x * TILE_SIZE + threadIdx.x;
    int target_pixel_height = blockIdx.y * TILE_SIZE + threadIdx.y;
    // printf("%d, %d\n", target_pixel_width, target_pixel_height);

    // printf("%f %f %f\n", d_camera_position.x, d_camera_position.y, d_camera_position.z);

    float3 delta_left = scalar_mult_float3(d_camera_left_direction, (target_pixel_width + 0.5 - RENDER_WIDTH / 2.0) * d_camera_pixel_width);
    float3 delta_up = scalar_mult_float3(d_camera_up_direction, (target_pixel_height + 0.5 - RENDER_HEIGHT / 2.0) * d_camera_pixel_height);
    float3 delta = add_float3(delta_left, add_float3(delta_up, scalar_mult_float3(d_camera_direction, d_camera_focal_length)));
    // float3 delta = make_float3((target_pixel_width + 0.5 - RENDER_WIDTH / 2.0) * d_camera_pixel_width, d_camera_focal_length, (target_pixel_height + 0.5 - RENDER_HEIGHT / 2.0) * d_camera_pixel_height);
    float3 pixel_center = make_float3(d_camera_position.x + delta.x, d_camera_position.y + delta.y, d_camera_position.z + delta.z);
    float pixel_radiance = ray_generation(pixel_center, curand_states);
    // float pixel_radiance = d_light_irradiance * curand_uniform(&curand_states[threadIdx.x]);

    // Gamma correction
    pixel_radiance /= d_light_irradiance;
    if (pixel_radiance > 1) {
        pixel_radiance = 1;
    }
    pixel_radiance = powf(pixel_radiance, 0.454545454545);

    
    unsigned char rgb_value = (unsigned char)(pixel_radiance * 255);
    // printf("%d, %d : %d\n", target_pixel_width, target_pixel_height, rgb_value);
    int base_idx = 3 * (target_pixel_height * RENDER_WIDTH + target_pixel_width);
    target_img[base_idx] = rgb_value;
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
    cudaMemcpyToSymbol(d_scene_objects, h_scene_objects, sizeof(h_scene_objects));

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