#include "gen2-whisper.h"
#include "whisper.h"
#include <cstring>
#include <cmath>
#include <new>
#include <mutex>
struct Check {g2w_cancel function;void * data;};
static bool cancelled(void * p){const auto * c=static_cast<Check *>(p);return c->function&&c->function(c->data)!=0;}
static bool encoding(whisper_context *,whisper_state *,void * p){return !cancelled(p);}
static void quiet(enum ggml_log_level,const char *,void *){}
static void * open_model(const void * model,size_t bytes){
    try {static std::once_flag logging;std::call_once(logging,[]{whisper_log_set(quiet,nullptr);});auto p=whisper_context_default_params();p.use_gpu=false;p.flash_attn=true;return whisper_init_from_buffer_with_params(const_cast<void *>(model),bytes,p);}catch(...){return nullptr;}
}
extern "C" {
const char * gen2_whisper_abi(void){return "gen2-whisper-v1:d1be6fde11ac6e0407606b4e42fe72d34add8037";}
void * gen2_whisper_open(const void * model,size_t bytes){
    if(!model||bytes!=59707625)return nullptr;
    return open_model(model,bytes);
}
void * gen2_whisper_open_tiny_v1(const void * model,size_t bytes){
    if(!model||bytes!=32152673)return nullptr;
    return open_model(model,bytes);
}
void gen2_whisper_close(void * context){if(context)whisper_free(static_cast<whisper_context *>(context));}
int gen2_whisper_run(void * context,const float * samples,size_t count,int language,int threads,g2w_cancel stop,void * user){
    if(!context||!samples||count<1800||count>480000||language<0||language>2||threads<1||threads>8)return 1;
    for(size_t i=0;i<count;i++)if(!std::isfinite(samples[i])||std::abs(samples[i])>1.0f)return 1;
    Check check{stop,user};if(cancelled(&check))return 2;
    try {
        auto p=whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        p.n_threads=threads;p.beam_search.beam_size=5;p.greedy.best_of=5;
        p.language=language==1?"en":language==2?"es":"auto";
        p.translate=false;p.no_context=true;p.no_timestamps=false;
        p.print_special=false;p.print_progress=false;p.print_realtime=false;p.print_timestamps=false;
        p.temperature=0.0f;p.temperature_inc=0.0f;
        p.abort_callback=cancelled;p.abort_callback_user_data=&check;
        p.encoder_begin_callback=encoding;p.encoder_begin_callback_user_data=&check;
        const int result=whisper_full(static_cast<whisper_context *>(context),p,samples,static_cast<int>(count));
        return cancelled(&check)?2:result==0?0:1;
    }catch(...){return 1;}
}
int gen2_whisper_segments(void * c){return c?whisper_full_n_segments(static_cast<whisper_context *>(c)):-1;}
int gen2_whisper_language(void * c){return c?whisper_full_lang_id(static_cast<whisper_context *>(c)):-1;}
int gen2_whisper_segment(void * c,int index,int64_t * start,int64_t * end,char * text,size_t capacity,size_t * written){
    if(!c||!start||!end||!text||!written||index<0||index>=gen2_whisper_segments(c))return 1;
    auto * ctx=static_cast<whisper_context *>(c);const char * value=whisper_full_get_segment_text(ctx,index);
    if(!value)return 1;
    const size_t count=std::strlen(value);if(count>capacity)return 1;
    *start=whisper_full_get_segment_t0(ctx,index);*end=whisper_full_get_segment_t1(ctx,index);
    std::memcpy(text,value,count);*written=count;return 0;
}
}
