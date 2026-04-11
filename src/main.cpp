// realesrgan implemented with ncnn library
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
#include <filesystem>
#include <string>
#if _WIN32
#include <locale>
#include <codecvt>
#endif

namespace fs = std::filesystem;

#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include "webp_image.h"
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb_image_resize2.h"

static const char *resizemodes[] = {
    "default",      // STBIR_FILTER_DEFAULT
    "box",          // STBIR_FILTER_BOX
    "triangle",     // STBIR_FILTER_TRIANGLE
    "cubicbspline", // STBIR_FILTER_CUBICBSPLINE
    "catmullrom",   // STBIR_FILTER_CATMULLROM
    "mitchell",     // STBIR_FILTER_MITCHELL
    "pointsample"   // STBIR_FILTER_POINT_SAMPLE
};

#if _WIN32
#include <wchar.h>
static wchar_t *optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t *const argv[], const wchar_t *optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t *p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t *optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t *p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}

static bool ascii_string_equals(const wchar_t *wide, const char *narrow)
{
    size_t widelen = wcslen(wide);
    size_t narrowlen = strlen(narrow);

    if (widelen != narrowlen)
        return false;

    for (size_t i = 0; i < widelen; i++)
    {
        if (wide[i] != narrow[i])
            return false;
    }

    return true;
}

static bool parse_optarg_resize(const wchar_t *optarg, int *width, int *height, int *mode, bool hasCustomWidth = false)
{
    *mode = 0; // default

    const wchar_t *colon = wcschr(optarg, L':');
    if (colon)
    {
        bool found = false;
        const wchar_t *modestr = colon + 1;
        for (int i = 0; i < (int)(sizeof(resizemodes) / sizeof(resizemodes[0])); i++)
        {
            if (ascii_string_equals(modestr, resizemodes[i]))
            {
                *mode = i;
                found = true;
                break;
            }
        }
        if (!found)
        {
            fwprintf(stderr, L"🚨 Error: Invalid resize mode '%s'\n", modestr);
            return false;
        }
    }

    if (hasCustomWidth)
    {
        return swscanf(optarg, L"%d", width) == 1;
    }

    return swscanf(optarg, L"%dx%d", width, height) == 2;
}

#else               // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char *optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char *p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}

static bool parse_optarg_resize(const char *optarg, int *width, int *height, int *mode, bool hasCustomWidth = false)
{
    *mode = 0; // default

    const char *colon = strchr(optarg, ':');
    if (colon)
    {
        bool found = false;
        const char *modestr = colon + 1;
        for (int i = 0; i < (int)(sizeof(resizemodes) / sizeof(resizemodes[0])); i++)
        {
            if (strcmp(modestr, resizemodes[i]) == 0)
            {
                *mode = i;
                found = true;
                break;
            }
        }
        if (!found)
        {
            fprintf(stderr, "🚨 Error: Invalid resize mode '%s'\n", modestr);
            return false;
        }
    }
    if (hasCustomWidth)
    {
        return sscanf(optarg, "%d", width) == 1;
    }
    return sscanf(optarg, "%dx%d", width, height) == 2;
}

#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "realesrgan.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: upscayl-bin -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -d                   enable daemon/interactive mode\n");
    fprintf(stderr, "  -z model-scale       scale according to the model (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -s output-scale      custom output scale (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -r resize            resize output to dimension (default=WxH:default), use '-r help' for more details\n");
    fprintf(stderr, "  -w width             resize output to a width (default=W:default), use '-r help' for more details\n");
    fprintf(stderr, "  -c compress          compression of the output image, default 0 and varies to 100\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -m model-path        folder path to the pre-trained models. default=models\n");
    fprintf(stderr, "  -n model-name        model name (default=realesrgan-x4plus, can be realesr-animevideov3 | realesrgan-x4plus-anime | realesrnet-x4plus or any other model)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x                   enable tta mode\n");
    fprintf(stderr, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  -v                   verbose output\n");
}

static void print_resize_usage()
{
    printf("'-r widthxheight:filter' argument usage:\n\n");

    printf("For example '-r 1920x1080' or '-r 1920x1080:default' will force all output images to be\n");
    printf("resized to 1920x1080 with the default filter if they aren't already.\n");
    printf("Similarly, '-w 1920' will force all output images to be resized to a width of 1920.\n\n");

    printf("Avaliable filters:\n");
    printf("  default       - Automatically decide\n");
    printf("  box           - A trapezoid w/1-pixel wide ramps, same result as box for integer scale ratios\n");
    printf("  triangle      - On upsampling, produces same results as bilinear texture filtering\n");
    printf("  cubicbspline  - The cubic b-spline (aka Mitchell-Netrevalli with B=1,C=0), gaussian-esque\n");
    printf("  catmullrom    - An interpolating cubic spline\n");
    printf("  mitchell      - Mitchell-Netrevalli filter with B=1/3, C=1/3\n");
    printf("  pointsample   - Simple point sampling\n");
}

static void print_daemon_help()
{
    fprintf(stderr, "\n📡 Daemon Mode Help\n");
    fprintf(stderr, "==================\n\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -s output-scale      custom output scale (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -r resize            resize output to dimension (default=WxH:default), use '-r help' for more details\n");
    fprintf(stderr, "  -w width             resize output to a width (default=W:default), use '-r help' for more details\n");
    fprintf(stderr, "  -c compress          compression of the output image, default 0 and varies to 100\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x                   enable tta mode\n");
    fprintf(stderr, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  help                 Show this help message\n");
    fprintf(stderr, "  quit or exit         Exit daemon mode\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  -i input.jpg -o output.jpg        Process a single image\n");
    fprintf(stderr, "  -i /path/to/dir -o /path/to/out   Process all images in a directory\n\n");
}

class Task
{
public:
    int id;
    int webp;
    bool outimage_malloced; // Flag to track if outimage.data was allocated with malloc

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task &v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task &v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int scale;
    int jobs_load;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
};

void *load(void *args)
{
    const LoadThreadParams *ltp = (const LoadThreadParams *)args;
    const int count = ltp->input_files.size();
    const int scale = ltp->scale;

#pragma omp parallel for schedule(static, 1) num_threads(ltp->jobs_load)
    for (int i = 0; i < count; i++)
    {
        const path_t &imagepath = ltp->input_files[i];

        int webp = 0;

        unsigned char *pixeldata = 0;
        int w;
        int h;
        int c;

#if _WIN32
        FILE *fp = _wfopen(imagepath.c_str(), L"rb");
#else
        FILE *fp = fopen(imagepath.c_str(), "rb");
#endif
        if (fp)
        {
            // read whole file
            unsigned char *filedata = 0;
            int length = 0;
            {
                fseek(fp, 0, SEEK_END);
                length = ftell(fp);
                rewind(fp);
                filedata = (unsigned char *)malloc(length);
                if (filedata)
                {
                    fread(filedata, 1, length, fp);
                }
                fclose(fp);
            }

            if (filedata)
            {
                pixeldata = webp_load(filedata, length, &w, &h, &c);
                if (pixeldata)
                {
                    webp = 1;
                }
                else
                {
                    // not webp, try jpg png etc.
#if _WIN32
                    pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
                    if (pixeldata)
                    {
                        // WIC channel conversion logic similar to stb_image
                        if (c == 1)
                        {
                            // grayscale -> rgb
                            unsigned char *rgbdata = (unsigned char *)malloc(w * h * 3);
                            if (rgbdata)
                            {
                                for (int i = 0; i < w * h; i++)
                                {
                                    unsigned char gray = pixeldata[i];
                                    rgbdata[i * 3 + 0] = gray; // B
                                    rgbdata[i * 3 + 1] = gray; // G
                                    rgbdata[i * 3 + 2] = gray; // R
                                }
                                free(pixeldata);
                                pixeldata = rgbdata;
                                c = 3;
                            }
                        }
                        else if (c == 2)
                        {
                            // grayscale + alpha -> rgba
                            unsigned char *rgbadata = (unsigned char *)malloc(w * h * 4);
                            if (rgbadata)
                            {
                                for (int i = 0; i < w * h; i++)
                                {
                                    unsigned char gray = pixeldata[i * 2];
                                    unsigned char alpha = pixeldata[i * 2 + 1];
                                    rgbadata[i * 4 + 0] = gray;  // B
                                    rgbadata[i * 4 + 1] = gray;  // G
                                    rgbadata[i * 4 + 2] = gray;  // R
                                    rgbadata[i * 4 + 3] = alpha; // A
                                }
                                free(pixeldata);
                                pixeldata = rgbadata;
                                c = 4;
                            }
                        }
                    }
#else  // _WIN32
                    pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
                    if (pixeldata)
                    {
                        // stb_image auto channel
                        if (c == 1)
                        {
                            // grayscale -> rgb
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
                            c = 3;
                        }
                        else if (c == 2)
                        {
                            // grayscale + alpha -> rgba
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 4);
                            c = 4;
                        }
                    }
#endif // _WIN32
                }

                free(filedata);
            }
        }
        if (pixeldata)
        {
            Task v;
            v.id = i;
            v.inpath = imagepath;
            v.outpath = ltp->output_files[i];
            v.outimage_malloced = false; // Initially managed by ncnn

            v.inimage = ncnn::Mat(w, h, (void *)pixeldata, (size_t)c, c);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

            path_t ext = get_file_extension(v.outpath);
            if (c == 4 && (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")))
            {
                path_t output_filename2 = get_file_name_without_extension(ltp->output_files[i]) + PATHSTR('.') + ext;
                v.outpath = output_filename2;
#if _WIN32
                fwprintf(stderr, L"ℹ️ Info: Image %s has alpha channel! Converting to RGB for JPEG output.\n", imagepath.c_str());
#else  // _WIN32
                fprintf(stderr, "ℹ️ Info: Image %s has alpha channel! Converting to RGB for JPEG output.\n", imagepath.c_str());
#endif // _WIN32
            }

            toproc.put(v);
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"🚨 Error: Couldn't read the image '%s'! (channels: %d)\n", imagepath.c_str(), c);
#else  // _WIN32
            fprintf(stderr, "🚨 Error: Couldn't read the image '%s'! (channels: %d)\n", imagepath.c_str(), c);
#endif // _WIN32
        }
    }

    return 0;
}

class ProcThreadParams
{
public:
    const RealESRGAN *realesrgan;
};

void *proc(void *args)
{
    const ProcThreadParams *ptp = (const ProcThreadParams *)args;
    const RealESRGAN *realesrgan = ptp->realesrgan;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        realesrgan->process(v.inimage, v.outimage);

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int resizeWidth;
    int resizeHeight;
    int resizeMode;
    bool resizeProvided;
    int outputScale;
    bool hasOutputScale;
    bool hasCustomWidth;
    float compression;
    int verbose;
};

void resize_output_image(Task &v, const SaveThreadParams *stp)
{
    const int resizeWidth = stp->resizeWidth;
    int resizeHeight = stp->resizeHeight;
    const bool resizeProvided = stp->resizeProvided;
    const bool hasCustomWidth = stp->hasCustomWidth;

    if ((!resizeProvided && !hasCustomWidth) ||
        (v.outimage.w == resizeWidth && v.outimage.h == resizeHeight) || (!resizeHeight && hasCustomWidth && v.outimage.w == resizeWidth))
    {
#if _WIN32
        fwprintf(stderr, L"⏩ Skipping resize\n");
#else  // _WIN32
        fprintf(stderr, "⏩ Skipping resize\n");
#endif // _WIN32
        return;
    }

    // Calculate the resize height if not provided
    if (hasCustomWidth)
    {
        resizeHeight = (v.inimage.h * resizeWidth) / v.inimage.w;
#if _WIN32
        fwprintf(stderr, L"🧮 Calculated height from width: %d\n", resizeHeight);
#else  // _WIN32
        fprintf(stderr, "🧮 Calculated height from width: %d\n", resizeHeight);
#endif // _WIN32
    }

#if _WIN32
    fwprintf(stderr, L"🏞️ Resizing image according to desired resolution\n");
#else  // _WIN32
    fprintf(stderr, "🏞️ Resizing image according to desired resolution\n");
#endif // _WIN32

    int c = v.outimage.elempack;

    stbir_pixel_layout layout = static_cast<stbir_pixel_layout>(c);

    // Create a new buffer for the resized image
    unsigned char *resizedData = (unsigned char *)malloc(resizeWidth * resizeHeight * c);

    // Resize the image using stb_image_resize
    stbir_resize_uint8_srgb((unsigned char *)v.outimage.data, v.outimage.w, v.outimage.h, 0, resizedData, resizeWidth, resizeHeight, 0, layout);

    // Free the old output image data only if it was malloc'd
    if (v.outimage_malloced && v.outimage.data)
    {
        free((void*)v.outimage.data);
    }

    // Replace the old image data with the new (resized) image data
    v.outimage = ncnn::Mat(resizeWidth, resizeHeight, resizedData, (size_t)c, c);
    v.outimage_malloced = true; // Now managed by malloc

#if _WIN32
    fwprintf(stderr, L"🏞️ Resized image from %dx%d to %dx%d\n", v.inimage.w, v.inimage.h, v.outimage.w, v.outimage.h);
#else  // _WIN32
    fprintf(stderr, "🏞️ Resized image from %dx%d to %dx%d\n", v.inimage.w, v.inimage.h, v.outimage.w, v.outimage.h);
#endif // _WIN32
}

void scale_output_image(Task &v, const SaveThreadParams *stp)
{
    const int originalWidth = v.inimage.w;
    const int originalHeight = v.inimage.h;
    const bool hasOutputScale = stp->hasOutputScale;
    const int outputScale = stp->outputScale;
    const int outputWidth = originalWidth * outputScale;
    const int outputHeight = originalHeight * outputScale;
    const bool resizeProvided = stp->resizeProvided;
    const bool hasCustomWidth = stp->hasCustomWidth;

    if (!hasOutputScale || resizeProvided || hasCustomWidth)
        return;

    int c = v.outimage.elempack;

#if _WIN32
    fwprintf(stderr, L"🏞️ Resizing image according to output scale\n");
#else  // _WIN32
    fprintf(stderr, "🏞️ Resizing image according to output scale\n");
#endif // _WIN32

    stbir_pixel_layout layout = static_cast<stbir_pixel_layout>(c);
    // Create a new buffer for the resized image
    unsigned char *resizedData = (unsigned char *)malloc(outputWidth * outputHeight * c);
    stbir_resize_uint8_srgb((unsigned char *)v.outimage.data, v.outimage.w, v.outimage.h, 0, resizedData, outputWidth, outputHeight, 0, layout);
    
    // Free the old output image data only if it was malloc'd
    if (v.outimage_malloced && v.outimage.data)
    {
        free((void*)v.outimage.data);
    }
    
    v.outimage = ncnn::Mat(outputWidth, outputHeight, resizedData, (size_t)v.outimage.elemsize, v.outimage.elemsize);
    v.outimage_malloced = true; // Now managed by malloc

#if _WIN32
    fwprintf(stderr, L"🏞️ Resized image from %dx%d to %dx%d\n", originalWidth, originalHeight, outputWidth, outputHeight);
#else  // _WIN32
    fprintf(stderr, "🏞️ Scaled image from %dx%d to %dx%d\n", originalWidth, originalHeight, outputWidth, outputHeight);
#endif // _WIN32
}

void *save(void *args)
{
    const SaveThreadParams *stp = (const SaveThreadParams *)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        // free input pixel data
        {
            unsigned char *pixeldata = (unsigned char *)v.inimage.data;
            if (v.webp == 1)
            {
                free(pixeldata);
            }
            else
            {
#if _WIN32
                free(pixeldata);
#else
                stbi_image_free(pixeldata);
#endif
            }
        }

        if (stp->hasOutputScale)
        {
            scale_output_image(v, stp);
        }

        if ((stp->resizeProvided || stp->hasCustomWidth) && !stp->hasOutputScale)
        {
            resize_output_image(v, stp);
        }

        int success = 0;

        path_t ext = get_file_extension(v.outpath);

        /* ----------- Create folder if not exists -------------------*/
        fs::path fs_path = fs::absolute(v.outpath);
#if _WIN32
        std::wstring parent_path = fs_path.parent_path().wstring();
#else
        std::string parent_path = fs_path.parent_path().string();
#endif

        if (!fs::exists(parent_path))
        {
            fprintf(stderr, "📂 Creating directory: %s\n", parent_path.c_str());
            fs::create_directories(parent_path);
        }

        if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char *)v.outimage.data, 100 - (int)stp->compression);
        }
        else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
#if _WIN32
            success = wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            // if compression is more than 0 make stbi_write_png_compression_level = 9
            if (stp->compression > 0)
            {
                stbi_write_png_compression_level = stp->compression;
            }
            else
            {
                stbi_write_png_compression_level = 9;
            }
            success = stbi_write_png(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 0);
#endif
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
#if _WIN32
            if (verbose)
            {
                fwprintf(stderr, L"🔧 Debug: Saving JPEG with %d channels, size %dx%d\n", v.outimage.elempack, v.outimage.w, v.outimage.h);
            }
            success = wic_encode_jpeg_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
            success = stbi_write_jpg(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 100 - (int)stp->compression);
#endif
        }
        if (success)
        {
            fprintf(stderr, "100.00%%\n");
            fprintf(stderr, "\n🙌 Upscayled Successfully!\n");

            if (verbose)
            {
#if _WIN32
                fwprintf(stderr, L"✅ %ls -> %ls done\n", v.inpath.c_str(), v.outpath.c_str());
#else
                fprintf(stderr, "✅ %s -> %s done\n", v.inpath.c_str(), v.outpath.c_str());
#endif
            }
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"🚨 Error: Couldn't write the image %s\n", v.outpath.c_str());
#else
            fprintf(stderr, "🚨 Error: Couldn't write the image %s\n", v.outpath.c_str());
#endif
        }
        
        // Free output image data only if it was allocated with malloc
        if (v.outimage_malloced && v.outimage.data)
        {
            free((void*)v.outimage.data);
        }
    }

    return 0;
}

struct ProcessParams
{
    int scale;
    int resizeWidth;
    int resizeHeight;
    int resizeMode;
    int outputScale;
    bool hasOutputScale;
    float compression;
    bool resizeProvided;
    bool hasCustomWidth;
    std::vector<int> tilesize;
    path_t model;
    path_t modelname;
    std::vector<int> gpuid;
    int jobs_load;
    std::vector<int> jobs_proc;
    int jobs_save;
    int verbose;
    int tta_mode;
    path_t format;
};

static ProcessParams create_process_params(
    int scale, int resizeWidth, int resizeHeight, int resizeMode,
    int outputScale, bool hasOutputScale, float compression,
    bool resizeProvided, bool hasCustomWidth,
    const std::vector<int> &tilesize, const path_t &model,
    const path_t &modelname, const std::vector<int> &gpuid,
    int jobs_load, const std::vector<int> &jobs_proc, int jobs_save,
    int verbose, int tta_mode, const path_t &format)
{
    ProcessParams params;
    params.scale = scale;
    params.resizeWidth = resizeWidth;
    params.resizeHeight = resizeHeight;
    params.resizeMode = resizeMode;
    params.outputScale = outputScale;
    params.hasOutputScale = hasOutputScale;
    params.compression = compression;
    params.resizeProvided = resizeProvided;
    params.hasCustomWidth = hasCustomWidth;
    params.tilesize = tilesize;
    params.model = model;
    params.modelname = modelname;
    params.gpuid = gpuid;
    params.jobs_load = jobs_load;
    params.jobs_proc = jobs_proc;
    params.jobs_save = jobs_save;
    params.verbose = verbose;
    params.tta_mode = tta_mode;
    params.format = format;
    return params;
}

static int process_image_batch(
    const path_t &inputpath,
    const path_t &outputpath,
    const ProcessParams &params,
    std::vector<RealESRGAN *> &realesrgan,
    int prepadding)
{
    // collect input and output filepath
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0)
                return -1;

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

            path_t last_filename;
            path_t last_filename_noext;
            for (int i = 0; i < count; i++)
            {
                path_t filename = filenames[i];
                path_t filename_noext = get_file_name_without_extension(filename);
                path_t output_filename = filename_noext + PATHSTR('.') + params.format;

                // filename list is sorted, check if output image path conflicts
                if (filename_noext == last_filename_noext)
                {
                    path_t output_filename2 = filename + PATHSTR('.') + params.format;
#if _WIN32
                    fwprintf(stderr, L"⚠️ Warning: both %s and %s output %s! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#else
                    fprintf(stderr, "⚠️ Warning: both %s and %s output %s! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#endif
                    output_filename = output_filename2;
                }
                else
                {
                    last_filename = filename;
                    last_filename_noext = filename_noext;
                }

                input_files[i] = inputpath + PATHSTR('/') + filename;
                output_files[i] = outputpath + PATHSTR('/') + output_filename;
            }
        }
        else if (!path_is_directory(inputpath) && !path_is_directory(outputpath))
        {
            input_files.push_back(inputpath);
            output_files.push_back(outputpath);
        }
        else
        {
            fprintf(stderr, "🚨 Error: Input path and Output path both must be either a file or a directory!\n");
            return -1;
        }
    }

    if (input_files.empty())
    {
        fprintf(stderr, "🚨 Error: No valid input files found!\n");
        return -1;
    }

    fprintf(stderr, "🚀 Processing %d image(s)...\n", (int)input_files.size());

    const int use_gpu_count = (int)params.gpuid.size();
    int total_jobs_proc = 0;
    for (int i = 0; i < use_gpu_count; i++)
    {
        total_jobs_proc += params.jobs_proc[i];
    }

    // main routine
    {
        // load image
        LoadThreadParams ltp;
        ltp.scale = params.scale;
        ltp.jobs_load = params.jobs_load;
        ltp.input_files = input_files;
        ltp.output_files = output_files;

        ncnn::Thread load_thread(load, (void *)&ltp);

        // realesrgan proc
        std::vector<ProcThreadParams> ptp(use_gpu_count);
        for (int i = 0; i < use_gpu_count; i++)
        {
            ptp[i].realesrgan = realesrgan[i];
        }

        std::vector<ncnn::Thread *> proc_threads(total_jobs_proc);
        {
            int total_jobs_proc_id = 0;
            for (int i = 0; i < use_gpu_count; i++)
            {
                for (int j = 0; j < params.jobs_proc[i]; j++)
                {
                    proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void *)&ptp[i]);
                }
            }
        }

        // save image
        SaveThreadParams stp;
        stp.resizeWidth = params.resizeWidth;
        stp.resizeHeight = params.resizeHeight;
        stp.resizeMode = params.resizeMode;
        stp.resizeProvided = params.resizeProvided;
        stp.verbose = params.verbose;
        stp.compression = params.compression;
        stp.outputScale = params.outputScale;
        stp.hasOutputScale = params.hasOutputScale;
        stp.hasCustomWidth = params.hasCustomWidth;

        std::vector<ncnn::Thread *> save_threads(params.jobs_save);
        for (int i = 0; i < params.jobs_save; i++)
        {
            save_threads[i] = new ncnn::Thread(save, (void *)&stp);
        }

        // end
        load_thread.join();

        Task end;
        end.id = -233;

        for (int i = 0; i < total_jobs_proc; i++)
        {
            toproc.put(end);
        }

        for (int i = 0; i < total_jobs_proc; i++)
        {
            proc_threads[i]->join();
            delete proc_threads[i];
        }

        for (int i = 0; i < params.jobs_save; i++)
        {
            tosave.put(end);
        }

        for (int i = 0; i < params.jobs_save; i++)
        {
            save_threads[i]->join();
            delete save_threads[i];
        }
    }

    return 0;
}

std::vector<std::string> split_cmdline(const std::string& cmd) {
    std::vector<std::string> args;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < cmd.size(); ++i) {
        char c = cmd[i];

        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (std::isspace(static_cast<unsigned char>(c)) && !in_quotes) {
            if (!current.empty()) {
                args.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }

    if (!current.empty())
        args.push_back(current);

    return args;
}

static int run_daemon_mode(ProcessParams &params)
{
    fprintf(stderr, "\n📡 Daemon Mode Started\n");

    ProcessParams originalParams;
    double compression = params.compression;
    int resizeWidth = params.resizeWidth;
    int resizeHeight = params.resizeHeight;
    int resizeMode = params.resizeMode;
    int jobs_load = params.jobs_load;
    int jobs_save = params.jobs_save;

    originalParams = params;

    // Load model once and keep it in memory
    int prepadding = 0;
    if (params.model.find(PATHSTR("models")) != path_t::npos || params.model.find(PATHSTR("models2")) != path_t::npos)
    {
        prepadding = 10;
    }
    else
    {
        fprintf(stderr, "🚨 Error: Unknown model dir type. Make sure that the model directory is called 'models' with *.param and *.bin files inside it.\n");
        return -1;
    }

#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];

    if (params.modelname == PATHSTR("realesr-animevideov3"))
    {
        swprintf(parampath, 256, L"%s/%s-x%d.param", params.model.c_str(), params.modelname.c_str(), params.scale);
        swprintf(modelpath, 256, L"%s/%s-x%d.bin", params.model.c_str(), params.modelname.c_str(), params.scale);
    }
    else
    {
        swprintf(parampath, 256, L"%s/%s.param", params.model.c_str(), params.modelname.c_str());
        swprintf(modelpath, 256, L"%s/%s.bin", params.model.c_str(), params.modelname.c_str());
    }
#else
    char parampath[256];
    char modelpath[256];

    if (params.modelname == PATHSTR("realesr-animevideov3"))
    {
        snprintf(parampath, sizeof(parampath), "%s/%s-x%d.param", params.model.c_str(), params.modelname.c_str(), params.scale);
        snprintf(modelpath, sizeof(modelpath), "%s/%s-x%d.bin", params.model.c_str(), params.modelname.c_str(), params.scale);
    }
    else
    {
        snprintf(parampath, sizeof(parampath), "%s/%s.param", params.model.c_str(), params.modelname.c_str());
        snprintf(modelpath, sizeof(modelpath), "%s/%s.bin", params.model.c_str(), params.modelname.c_str());
    }
#endif

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

    const int use_gpu_count = (int)params.gpuid.size();
    std::vector<RealESRGAN *> realesrgan(use_gpu_count);

    fprintf(stderr, "🔧 Loading model...\n");
    for (int i = 0; i < use_gpu_count; i++)
    {
        realesrgan[i] = new RealESRGAN(params.gpuid[i], params.tta_mode);
        realesrgan[i]->load(paramfullpath, modelfullpath);
        realesrgan[i]->scale = params.scale;
        realesrgan[i]->tilesize = params.tilesize[i];
        realesrgan[i]->prepadding = prepadding;
    }
    fprintf(stderr, "✅ Model loaded successfully!\n");
#if _WIN32
    fwprintf(stderr, L"Model: %s (scale: %dx)\n", params.modelname.c_str(), params.scale);
#else
    fprintf(stderr, "Model: %s (scale: %dx)\n", params.modelname.c_str(), params.scale);
#endif
    fprintf(stderr, "Type 'help' for usage or 'quit' to exit\n\n");

    std::string line;
    while (true)
    {
        fprintf(stderr, "📡 Ready> ");

#if _WIN32
        std::wstring wline;
        if (!std::getline(std::wcin, wline))
        {
            break; // EOF or error
        }

        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv1;
        line = conv1.to_bytes(wline);
#else
        if (!std::getline(std::cin, line))
        {
            break; // EOF or error
        }
#endif

        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos)
        {
            continue; // Empty line
        }
        line = line.substr(start, end - start + 1);

        if (line.empty())
        {
            continue;
        }

        if (line == "quit" || line == "exit")
        {
            fprintf(stderr, "👋 Exiting daemon mode...\n");
            break;
        }

        if (line == "help")
        {
            print_daemon_help();
            continue;
        }

        params = originalParams;

        auto tokens = split_cmdline(line);

        std::vector<std::string> parsed_args;
        parsed_args.emplace_back("upscayl");
        parsed_args.insert(parsed_args.end(), tokens.begin(), tokens.end());

        std::vector<char*> argv;
        for (auto& a : parsed_args)
            argv.push_back(a.data());
        argv.push_back(nullptr);

        int argc = static_cast<int>(argv.size() - 1);

        // Reset getopt global state
        optind = 1;
        optarg = NULL;

#if _WIN32
        std::wstring input_str, output_str;
        wchar_t opt;

        std::vector<std::wstring> wargs;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv2;

        for (int i = 0; i < argc; i++)
        {
            wargs.push_back(conv2.from_bytes(argv[i]));
        }
        
        std::vector<wchar_t*> wargv;
        for (auto& a : wargs)
        {
            wargv.push_back(a.data());
        }
        wargv.push_back(nullptr);
        while ((opt = getopt(argc, wargv.data(), L"i:o:s:r:w:t:c:j:f:x")) != (wchar_t)-1)
        {
            switch (opt)
            {
            case L'i':
                input_str = optarg;
                break;
            case L'o':
                output_str = optarg;
                break;
            case L's':
                params.outputScale = _wtoi(optarg);
                params.hasOutputScale = true;
                break;
            case L'c':
                compression = _wtof(optarg);
                if (compression < 0 || compression > 100)
                {
                    fwprintf(stderr, L"🚨 Error: Invalid compression value, it should be between 0 and 100!\n");
                    return -1;
                }
                params.compression = round(compression / 10.0f) * 10.0f;
                break;
            case L'r':
                if (wcscmp(optarg, L"help") == 0)
                {
                    print_resize_usage();
                    return -1;
                }
                if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode))
                {
                    fwprintf(stderr, L"🚨 Error: Invalid resize value!\n");
                    return -1;
                }
                params.resizeProvided = true;
                params.resizeWidth = resizeWidth;
                params.resizeHeight = resizeHeight;
                params.resizeMode = resizeMode;
                break;
            case L'w':
                if (wcscmp(optarg, L"help") == 0)
                {
                    print_resize_usage();
                    return -1;
                }
                if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode, true))
                {
                    fwprintf(stderr, L"🚨 Error: Invalid resize value!\n");
                    return -1;
                }
                params.hasCustomWidth = true;
                params.resizeWidth = resizeWidth;
                params.resizeHeight = resizeHeight;
                params.resizeMode = resizeMode;
                break;
            case L't':
                params.tilesize = parse_optarg_int_array(optarg);
                break;
            case L'j':
                swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
                params.jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
                params.jobs_load = jobs_load;
                params.jobs_save = jobs_save;
                break;
            case L'f':
                params.format = optarg;
                break;
            case L'x':
                params.tta_mode = 1;
                break;
            }
        }
#else
        std::string input_str, output_str;
        int opt;
        while ((opt = getopt(argc, argv.data(), "i:o:s:r:w:t:c:j:f:x")) != -1)
        {
            switch (opt)
            {
            case 'i':
                input_str = optarg;
                break;
            case 'o':
                output_str = optarg;
                break;
            case 's':
                params.outputScale = atoi(optarg);
                params.hasOutputScale = true;
                break;
            case 'c':
                compression = atof(optarg);
                if (compression < 0 || compression > 100)
                {
                    fprintf(stderr, "🚨 Error: Invalid compression value, it should be between 0 and 100!\n");
                    return -1;
                }
                params.compression = round(compression / 10.0) * 10;
                break;
            case 'r':
                if (strcmp(optarg, "help") == 0)
                {
                    print_resize_usage();
                    return -1;
                }
                if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode))
                {
                    fprintf(stderr, "🚨 Error: Invalid resize value!\n");
                    return -1;
                }
                params.resizeProvided = true;
                params.resizeWidth = resizeWidth;
                params.resizeHeight = resizeHeight;
                params.resizeMode = resizeMode;
                break;
            case 'w':
                if (strcmp(optarg, "help") == 0)
                {
                    print_resize_usage();
                    return -1;
                }
                if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode, true))
                {
                    fprintf(stderr, "🚨 Error: Invalid resize value!\n");
                    return -1;
                }
                params.hasCustomWidth = true;
                params.resizeWidth = resizeWidth;
                params.resizeHeight = resizeHeight;
                params.resizeMode = resizeMode;
                break;
            case 't':
                params.tilesize = parse_optarg_int_array(optarg);
                break;
            case 'j':
                sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
                params.jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
                params.jobs_load = jobs_load;
                params.jobs_save = jobs_save;
                break;
            case 'f':
                params.format = optarg;
                break;
            case 'x':
                params.tta_mode = 1;
                break;
            }
        }
#endif
        
        if (input_str.empty() || output_str.empty())
        {
            fprintf(stderr, "🚨 Error: Both input and output paths are required.\n\n");
            continue;
        }

        path_t inputpath = input_str;
        path_t outputpath = output_str;

        // Process the images with pre-loaded model
        int result = process_image_batch(inputpath, outputpath, params, realesrgan, prepadding);
        if (result != 0)
        {
            fprintf(stderr, "❌ Processing failed\n\n");
        }
        else
        {
            fprintf(stderr, "\n");
        }
    }

    // Clean up models
    for (int i = 0; i < use_gpu_count; i++)
    {
        delete realesrgan[i];
    }

    return 0;
}

#if _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
    setlocale(LC_ALL, "");
    path_t inputpath;
    path_t outputpath;
    int scale = 4;
    int resizeWidth;
    int resizeHeight;
    int resizeMode;
    int outputScale = 4;
    bool hasOutputScale = false;
    float compression = 0.00f;
    bool resizeProvided = false;
    bool hasCustomWidth = false;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models");
    path_t modelname = PATHSTR("realesrgan-x4plus");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");
    bool daemon_mode = false;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    fprintf(stderr, "🚀 Starting Upscayl - Copyright © 2024\n");
    while ((opt = getopt(argc, argv, L"i:o:z:s:r:w:t:c:m:n:g:j:f:vxhd")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L'z':
            scale = _wtoi(optarg);
            break;
        case L's':
            outputScale = _wtoi(optarg);
            hasOutputScale = true;
            break;
        case L'c':
            compression = _wtof(optarg);
            if (compression < 0 || compression > 100)
            {
                fwprintf(stderr, L"🚨 Error: Invalid compression value, it should be between 0 and 100!\n");
                return -1;
            }
            compression = round(compression / 10.0) * 10;
            break;
        case L'r':
            if (wcscmp(optarg, L"help") == 0)
            {
                print_resize_usage();
                return -1;
            }
            if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode))
            {
                fwprintf(stderr, L"🚨 Error: Invalid resize value!\n");
                return -1;
            }
            resizeProvided = true;
            break;
        case L'w':
            if (wcscmp(optarg, L"help") == 0)
            {
                print_resize_usage();
                return -1;
            }
            if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode, true))
            {
                fwprintf(stderr, L"🚨 Error: Invalid resize value!\n");
                return -1;
            }
            hasCustomWidth = true;
            break;
        case L't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'n':
            modelname = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'f':
            format = optarg;
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'd':
            daemon_mode = true;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else  // _WIN32
    int opt;
    fprintf(stderr, "🚀 Starting Upscayl - Copyright © 2024\n");
    while ((opt = getopt(argc, argv, "i:o:z:s:r:w:t:c:m:n:g:j:f:vxhd")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'z':
            scale = atoi(optarg);
            break;
        case 's':
            outputScale = atoi(optarg);
            hasOutputScale = true;
            break;
        case 'c':
            compression = atof(optarg);
            if (compression < 0 || compression > 100)
            {
                fprintf(stderr, "🚨 Error: Invalid compression value, it should be between 0 and 100!\n");
                return -1;
            }
            compression = round(compression / 10.0) * 10;
            break;
        case 'r':
            if (strcmp(optarg, "help") == 0)
            {
                print_resize_usage();
                return -1;
            }
            if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode))
            {
                fprintf(stderr, "🚨 Error: Invalid resize value!\n");
                return -1;
            }
            resizeProvided = true;
            break;
        case 'w':
            if (strcmp(optarg, "help") == 0)
            {
                print_resize_usage();
                return -1;
            }
            if (!parse_optarg_resize(optarg, &resizeWidth, &resizeHeight, &resizeMode, true))
            {
                fprintf(stderr, "🚨 Error: Invalid resize value!\n");
                return -1;
            }
            hasCustomWidth = true;
            break;
        case 't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'n':
            modelname = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'd':
            daemon_mode = true;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32
    if (!daemon_mode && (inputpath.empty() || outputpath.empty()))
    {
        print_usage();
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "🚨 Error: Invalid tile size!\n");
        return -1;
    }

    for (int i = 0; i < (int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "🚨 Error: Invalid tile size!\n");
            return -1;
        }
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "🚨 Error: Invalid thread count!\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "🚨 Error: invalid jobs_proc thread count!\n");
        return -1;
    }

    for (int i = 0; i < (int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "🚨 Error: Invalid jobs_proc thread count argument!\n");
            return -1;
        }
    }

    if (!daemon_mode && !path_is_directory(outputpath))
    {
        path_t ext = format;

        if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
            format = PATHSTR("png");
        }
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            format = PATHSTR("webp");
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            format = PATHSTR("jpg");
        }
        else
        {
            fprintf(stderr, "🚨 Error: Invalid output path extension or type!\n");
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
    {
        fprintf(stderr, "🚨 Error: Invalid format provided!\n");
        return -1;
    }

    // collect input and output filepath (skip in daemon mode)
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    if (!daemon_mode)
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0)
                return -1;

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

            path_t last_filename;
            path_t last_filename_noext;
            for (int i = 0; i < count; i++)
            {
                path_t filename = filenames[i];
                path_t filename_noext = get_file_name_without_extension(filename);
                path_t output_filename = filename_noext + PATHSTR('.') + format;

                // filename list is sorted, check if output image path conflicts
                if (filename_noext == last_filename_noext)
                {
                    path_t output_filename2 = filename + PATHSTR('.') + format;
#if _WIN32
                    fwprintf(stderr, L"⚠️ Warning: both %s and %s output %s! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#else
                    fprintf(stderr, "⚠️ Warning: both %s and %s output %s! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#endif
                    output_filename = output_filename2;
                }
                else
                {
                    last_filename = filename;
                    last_filename_noext = filename_noext;
                }

                input_files[i] = inputpath + PATHSTR('/') + filename;
                output_files[i] = outputpath + PATHSTR('/') + output_filename;
            }
        }
        else if (!path_is_directory(inputpath) && !path_is_directory(outputpath))
        {
            input_files.push_back(inputpath);
            output_files.push_back(outputpath);
        }
        else
        {
            fprintf(stderr, "🚨 Error: Input path and Output path both must be either a file or a directory!\n");
            return -1;
        }
    }

    int prepadding = 0;

    if (model.find(PATHSTR("models")) != path_t::npos || model.find(PATHSTR("models2")) != path_t::npos)
    {
        prepadding = 10;
    }
    else
    {
        fprintf(stderr, "🚨 Error: Unknown model dir type. Make sure that the model directory is called 'models' with *.param and *.bin files inside it.\n");
        return -1;
    }

    // if (modelname.find(PATHSTR("realesrgan-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("realesrnet-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("esrgan-x4")) != path_t::npos)
    // {}
    // else
    // {
    //     fprintf(stderr, "unknown model name\n");
    //     return -1;
    // }

#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        swprintf(parampath, 256, L"%s/%s-x%d.param", model.c_str(), modelname.c_str(), scale);
        swprintf(modelpath, 256, L"%s/%s-x%d.bin", model.c_str(), modelname.c_str(), scale);
    }
    else
    {
        swprintf(parampath, 256, L"%s/%s.param", model.c_str(), modelname.c_str());
        swprintf(modelpath, 256, L"%s/%s.bin", model.c_str(), modelname.c_str());
    }

#else
    char parampath[256];
    char modelpath[256];

    // Check if modelname includes scale
    if (modelname.find(PATHSTR("x1")) != path_t::npos || modelname.find(PATHSTR("1x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x1\n");
        scale = 1;
    }
    else if (modelname.find(PATHSTR("x2")) != path_t::npos || modelname.find(PATHSTR("2x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x2\n");
        scale = 2;
    }
    else if (modelname.find(PATHSTR("x3")) != path_t::npos || modelname.find(PATHSTR("3x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x3\n");
        scale = 3;
    }
    else if (modelname.find(PATHSTR("x4")) != path_t::npos || modelname.find(PATHSTR("4x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x4\n");
        scale = 4;
    }
    else if (modelname.find(PATHSTR("x8")) != path_t::npos || modelname.find(PATHSTR("8x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x8\n");
        scale = 8;
    }
    else if (modelname.find(PATHSTR("x16")) != path_t::npos || modelname.find(PATHSTR("16x")) != path_t::npos)
    {
        fprintf(stderr, "✨ Detected scale x16\n");
        scale = 16;
    }

    if (scale == 4)
    {
        fprintf(stderr, "✨ Using the default scale x4\n");
    }

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        snprintf(parampath, sizeof(parampath), "%s/%s-x%d.param", model.c_str(), modelname.c_str(), scale);
        snprintf(modelpath, sizeof(modelpath), "%s/%s-x%d.bin", model.c_str(), modelname.c_str(), scale);
    }
    else
    {
        snprintf(parampath, sizeof(parampath), "%s/%s.param", model.c_str(), modelname.c_str());
        snprintf(modelpath, sizeof(modelpath), "%s/%s.bin", model.c_str(), modelname.c_str());
    }
#endif

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i = 0; i < use_gpu_count; i++)
    {
        if (gpuid[i] < 0 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "🚨 Error: Invalid GPU Device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i = 0; i < use_gpu_count; i++)
    {
        int gpu_queue_count = ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    for (int i = 0; i < use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 200;
            else if (heap_budget > 550)
                tilesize[i] = 100;
            else if (heap_budget > 190)
                tilesize[i] = 64;
            else
                tilesize[i] = 32;
        }
    }

    // Branch between daemon mode and single-run mode
    if (daemon_mode)
    {
        // Prepare parameters for daemon mode
        ProcessParams params = create_process_params(
            scale, resizeWidth, resizeHeight, resizeMode,
            outputScale, hasOutputScale, compression,
            resizeProvided, hasCustomWidth, tilesize, model,
            modelname, gpuid, jobs_load, jobs_proc, jobs_save,
            verbose, tta_mode, format);

        int result = run_daemon_mode(params);
        
        ncnn::destroy_gpu_instance();
        return result;
    }

    // Single-run mode: process the specified input/output paths
    {
        // Prepare parameters
        ProcessParams params = create_process_params(
            scale, resizeWidth, resizeHeight, resizeMode,
            outputScale, hasOutputScale, compression,
            resizeProvided, hasCustomWidth, tilesize, model,
            modelname, gpuid, jobs_load, jobs_proc, jobs_save,
            verbose, tta_mode, format);

        std::vector<RealESRGAN *> realesrgan(use_gpu_count);

        for (int i = 0; i < use_gpu_count; i++)
        {
            realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);
            realesrgan[i]->load(paramfullpath, modelfullpath);
            realesrgan[i]->scale = scale;
            realesrgan[i]->tilesize = tilesize[i];
            realesrgan[i]->prepadding = prepadding;
        }

        int result = process_image_batch(inputpath, outputpath, params, realesrgan, prepadding);

        for (int i = 0; i < use_gpu_count; i++)
        {
            delete realesrgan[i];
        }
        realesrgan.clear();

        if (result != 0)
        {
            ncnn::destroy_gpu_instance();
            return result;
        }
    }

    ncnn::destroy_gpu_instance();

    return 0;
}