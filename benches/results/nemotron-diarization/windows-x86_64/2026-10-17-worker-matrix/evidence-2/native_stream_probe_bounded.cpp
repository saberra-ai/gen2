// Valid-Mel V1 reference using the pinned installed public C header.
// Historical native_stream_probe.cpp remains unchanged.
// Qualification only: bounded licensed WAV repeated through one native session.
#include "nemo_speech/diar.h"
#include "wav_reader.h"
#include "sha256/sha256.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string_view>
#include <vector>

using Clock = std::chrono::steady_clock;
static uint64_t micros(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
}
static void check(nemo_speech_asr_status status) {
    if (status != NEMO_SPEECH_ASR_OK) throw std::runtime_error(nemo_speech_asr_last_error());
}
static int number(std::string_view text) {
    int value=0;
    const auto result=std::from_chars(text.data(),text.data()+text.size(),value);
    if(result.ec!=std::errc{} || result.ptr!=text.data()+text.size()) throw std::runtime_error("invalid integer argument");
    return value;
}
// Use the SHA256 source from the pinned llama.cpp/vendor/hash tree. Hash the
// actual normalized float vector supplied to the C API, in portable LE order.
static std::string input_digest(const std::vector<float>& audio,int repeats) {
    static_assert(sizeof(float)==4);
    sha256_t state;sha256_init(&state);
    for(int repeat=0;repeat<repeats;repeat++)for(float value:audio) {
        uint32_t bits;std::memcpy(&bits,&value,4);
        const unsigned char bytes[]={static_cast<unsigned char>(bits),static_cast<unsigned char>(bits>>8),
            static_cast<unsigned char>(bits>>16),static_cast<unsigned char>(bits>>24)};
        sha256_update(&state,bytes,4);
    }
    unsigned char hash[SHA256_DIGEST_SIZE];sha256_final(&state,hash);
    std::ostringstream result;result<<std::hex<<std::setfill('0');
    for(unsigned char byte:hash)result<<std::setw(2)<<static_cast<unsigned>(byte);
    return result.str();
}
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("usage: native_stream_probe MODEL WAV REPEATS GPU_INDEX OUTPUT.json");
        const std::string probability_path=std::string(argv[5])+".f32";
        if(std::filesystem::exists(argv[5]) || std::filesystem::exists(probability_path))
            throw std::runtime_error("refusing to replace existing native evidence");
        const char* version=nemo_speech_asr_version();
        if(!version || std::strcmp(version,"nemo-speech-asr 0.2.0+gen2-diagnostic-cpu-threads-2"))
            throw std::runtime_error("unexpected valid-Mel runtime profile");
        const int repeats=number(argv[3]),gpu=number(argv[4]);
        if(repeats<=0 || gpu < -1)throw std::runtime_error("positive repeats and GPU index>=-1 required");
        if(std::filesystem::file_size(argv[2])>20*1024*1024)throw std::runtime_error("use a WAV fixture no larger than 20 MiB");
        std::vector<float> audio;
        std::string error;
        if(!examples::read_wav_mono_16k(argv[2],audio,error))throw std::runtime_error(error);
        const uint64_t samples=uint64_t(audio.size())*uint64_t(repeats);
        if(audio.size()<320 || audio.size()>600*16000 || samples>3600ULL*16000
            || !std::all_of(audio.begin(),audio.end(),[](float x){return std::isfinite(x);}))
            throw std::runtime_error("use finite 20 ms–600 s mono 16 kHz audio totaling at most one hour");
        const auto input_sha256=input_digest(audio,repeats);
        nemo_speech_diar_model_config config{};
        config.size=sizeof(config);config.model_path=argv[1];config.gpu=gpu;
        config.preset="v3-streaming";config.chunk_frames=9;config.right_context_frames=4;
        config.left_context_frames=0;config.fifo_frames=264;config.spkcache_frames=264;config.update_period_frames=222;
        auto started=Clock::now();
        nemo_speech_diar_model* raw_model=nullptr;
        check(nemo_speech_diar_create(&config,&raw_model));
        std::unique_ptr<nemo_speech_diar_model,decltype(&nemo_speech_diar_destroy)> model(raw_model,nemo_speech_diar_destroy);
        const auto load_micros=micros(started);
        if(nemo_speech_diar_num_speakers(raw_model)!=8 || std::abs(nemo_speech_diar_seconds_per_frame(raw_model)-.01)>1e-9)
            throw std::runtime_error("expected V3 eight-speaker10ms model");
        nemo_speech_diar_stream* raw_stream=nullptr;
        check(nemo_speech_diar_stream_open(raw_model,&raw_stream));
        std::unique_ptr<nemo_speech_diar_stream,decltype(&nemo_speech_diar_stream_close)> stream(raw_stream,nemo_speech_diar_stream_close);
        std::vector<uint64_t> push_times;
        started=Clock::now();
        for(int repeat=0;repeat<repeats;repeat++) {
            for(size_t offset=0;offset<audio.size();offset+=16640) {
                const auto push_start=Clock::now();
                check(nemo_speech_diar_stream_push_f32(raw_stream,audio.data()+offset,std::min(size_t(16640),audio.size()-offset),16000));
                push_times.push_back(micros(push_start));
            }
        }
        const auto finish_start=Clock::now();check(nemo_speech_diar_stream_finish(raw_stream));
        const auto finish_micros=micros(finish_start),elapsed_micros=micros(started);
        const auto frames=nemo_speech_diar_frame_count(raw_stream),first=nemo_speech_diar_frame_probs_start(raw_stream);
        if(frames<0 || frames>int64_t(samples/160+1) || first<0 || first>frames)throw std::runtime_error("unexpected native frame bounds");
        if(frames-first>130000)throw std::runtime_error("excessive retained native probabilities");
        std::vector<float> probabilities(size_t(frames-first)*8);
        check(nemo_speech_diar_frame_probs(raw_stream,probabilities.data(),probabilities.size()));
        std::ofstream probability_output(probability_path,std::ios::binary);
        probability_output.exceptions(std::ios::failbit|std::ios::badbit);
        probability_output.write(reinterpret_cast<const char*>(probabilities.data()),probabilities.size()*sizeof(float));
        probability_output.close();
        if(frames!=int64_t(samples/160))throw std::runtime_error("unexpected final valid-Mel frame count");
        if(!std::all_of(probabilities.begin(),probabilities.end(),[](float x){return std::isfinite(x)&&x>=0&&x<=1;}))
            throw std::runtime_error("invalid native probabilities");
        size_t count=0;check(nemo_speech_diar_segments(raw_stream,nullptr,nullptr,0,&count));
        if(count>100000)throw std::runtime_error("excessive native turns");
        std::vector<nemo_speech_diar_segment> turns(count);
        if(count)check(nemo_speech_diar_segments(raw_stream,nullptr,turns.data(),count,&count));
        uint64_t total_push_micros=0;for(auto t:push_times)total_push_micros+=t;
        std::sort(push_times.begin(),push_times.end());
        const auto percentile=[&](double p){return push_times.at(size_t(std::ceil(p*push_times.size()))-1);};
        std::ofstream output(argv[5]);output.exceptions(std::ios::failbit|std::ios::badbit);
        output<<std::setprecision(17)<<"{\"audio_samples\":"<<samples<<",\"audio_duration_seconds\":"<<samples/16000.
          <<",\"runtime_version\":\""<<version<<"\",\"input_pcm_f32le_sha256\":\""<<input_sha256<<"\",\"input_repeats\":"<<repeats
          <<",\"geometry\":{\"chunk\":9,\"right\":4,\"left\":0,\"fifo\":264,\"cache\":264,\"update\":222}"
          <<",\"requested_gpu_index\":"<<gpu<<",\"cold_native_load_micros\":"<<load_micros
          <<",\"push_count\":"<<push_times.size()<<",\"native_push_median_micros\":"<<percentile(.5)
          <<",\"native_push_p95_micros\":"<<percentile(.95)<<",\"native_push_max_micros\":"<<push_times.back()
          <<",\"native_push_total_micros\":"<<total_push_micros<<",\"finish_micros\":"<<finish_micros
          <<",\"elapsed_micros\":"<<elapsed_micros<<",\"native_frame_count\":"<<frames
          <<",\"retained_first_frame\":"<<first<<",\"retained_rows\":"<<frames-first<<",\"turns\":[";
        for(size_t i=0;i<count;i++) {
            const auto& t=turns[i];
            if(!std::isfinite(t.start_time)||!std::isfinite(t.end_time)||t.start_time<0||t.end_time<t.start_time||t.speaker<1||t.speaker>8)
                throw std::runtime_error("invalid native turn");
            if(i)output<<',';
            output<<"{\"start_seconds\":"<<t.start_time<<",\"end_seconds\":"<<t.end_time<<",\"speaker\":"<<t.speaker<<'}';
        }
        output<<"]}\n";output.close();return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
