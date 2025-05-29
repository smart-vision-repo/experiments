#include <iostream>
extern "C" {
#include <libavcodec/avcodec.h>
}
int main() {
    AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec) {
        std::cout << "Decoder found: " << codec->name << std::endl;
    } else {
        std::cout << "Decoder not found" << std::endl;
    }
}

