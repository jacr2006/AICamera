#include <jni.h>
#include <string>
#include <algorithm>
#include <ctime>

#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1

#include <torch/script.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>


#include <android/log.h>
#include <ATen/core/ArrayRef.h>
#include <ATen/ArrayRef.h>
//#include <arm_neon.h>

#define IMG_H 227
#define IMG_W 227
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);

static std::shared_ptr<torch::jit::script::Module> module;
static float input_data[MAX_DATA_SIZE];

void load_model(AAssetManager* mgr, const char* fn) {
    AAsset* asset = AAssetManager_open(mgr, "traced_model.pt", AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    std::string asset_s((const char *) data, len);
    {
        auto stream = std::istringstream(asset_s);
        module = torch::jit::load(stream);
    }
    assert(module != nullptr);
    AAsset_close(asset);
}

static double now_seconds(void) {
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return res.tv_sec + (double) res.tv_nsec * 1e-9;
}


extern "C"
void
Java_mathinf_neustyle_ClassifyCamera_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject jAssetManager) {
    alog("Loading model");
    auto mgr = AAssetManager_fromJava(env, jAssetManager);
    load_model(mgr, "model_traced.pt");
    alog("done.")
}

static at::Tensor transform_tensor(at::Tensor& input) {
    at::Tensor output;
    if (module != nullptr) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        output = module->forward(inputs).toTensor();
    }
    return output;
}

extern "C"
JNIEXPORT jstring JNICALL Java_mathinf_neustyle_ClassifyCamera_torchTransform(JNIEnv * env, jobject  obj, jobject bitmap,
                                                                          jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
                                                                          jint yRowStride, jint rowStride, jint pixelStride, jint sensorOrientation) {
    AndroidBitmapInfo  info;
    void*              pixels;
    int                ret;
    static int         init;

    if (!init) {
        //init_tables();
        //stats_init(&stats);
        init = 1;
    }

    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        alog("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return env->NewStringUTF("error bitmap");
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGB_565) {
        alog("Bitmap format is not RGB_565 !");
        return env->NewStringUTF("error bitmap2");
    }

    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        alog("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return env->NewStringUTF("error bitmap3");
    }


    if (module == nullptr) {
        return env->NewStringUTF("Loading X...");
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    // poor person's center crop
    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    int cosr = 1;
    int sinr = 0;
    int roti_offset = 0;
    int rotj_offset = 0;

    if (sensorOrientation == 90) {
        cosr = 0;
        sinr = 1;
        rotj_offset = iter_h - 1;
    } else if (sensorOrientation == 180) {
        cosr = -1;
        sinr = 0;
        roti_offset = iter_h - 1;
        rotj_offset = iter_w - 1;
    } else if (sensorOrientation == 270) {
        cosr = 0;
        sinr = -1;
        roti_offset = iter_w - 1;
    }

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * yRowStride];
        jbyte* U_row = &U_data[((h_offset + i)/2) * rowStride]; // strides for U V are guaranteed to be the same, YUV_420 means that the vertical resolution of u/v is subsampled by a factor two
        jbyte* V_row = &V_data[((h_offset + i)/2) * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            uint8_t y = Y_row[w_offset + j] & 0xff; // pixelStride is guaranteed to be 1
            uint8_t u = U_row[pixelStride * ((w_offset+j)/2)] & 0xff;
            uint8_t v = V_row[pixelStride * ((w_offset+j)/2)] & 0xff;

            int roti = roti_offset + cosr * i + sinr * j;
            int rotj = rotj_offset - sinr * i + cosr * i;
            // this relies on IMG_H == IMG_W to not risk bad things!

            auto r_i = 0 * IMG_H * IMG_W + roti * IMG_W + rotj;
            auto g_i = 1 * IMG_H * IMG_W + roti * IMG_W + rotj;
            auto b_i = 2 * IMG_H * IMG_W + roti * IMG_W + rotj;

            input_data[r_i] = (float) ((float) min(255., max(0., (float) (y + 1.370705 * ((float) v - 128)))));
            input_data[g_i] = (float) ((float) min(255., max(0., (float) (y - 0.337633 * ((float) u - 128) - 0.698001 * ((float)v - 128)))));
            input_data[b_i] = (float) ((float) min(255., max(0., (float) (y + 1.732446 * ((float) u - 128)))));
        }
    }

    auto input_ = torch::tensor(at::ArrayRef<float>(input_data, MAX_DATA_SIZE));
    auto input = input_.view({1, IMG_C, IMG_H, IMG_W })/255;

    std::string s;
    at::Tensor output_;
    double duration = -now_seconds();
    try {
        output_ = transform_tensor(input).clamp(0, 1.0);
    } catch (std::runtime_error e) {
        s = e.what();
    }
    duration += now_seconds();

    if (output_.defined()) {
        auto output = output_.accessor<float, 4>()[0];
        for (int yy = 0; yy < info.height; yy++) {
            uint16_t *line = (uint16_t *) pixels;
            for (int xx = 0; xx < info.width; xx++) {
                line[xx] = (static_cast<uint16_t>(output[0][yy][xx] * 0x1f + 0.499) << 11)
                           + (static_cast<uint16_t>(output[1][yy][xx] * 0x3f + 0.499) << 5)
                           + (static_cast<uint16_t>(output[2][yy][xx] * 0x1f + 0.499));
            }

            // go to next line
            pixels = (char *) pixels + info.stride;
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);

    if (! s.empty()) {
        return env->NewStringUTF(s.c_str());
    }
    std::stringstream stream;
    stream << "Neural time: " << duration << "s";
    return env->NewStringUTF(stream.str().c_str());
}

