/* core/server/server.h -- the OpenAI-compatible server (spec §14).
 *
 * Threading (spec §1): HTTP threads feed a request queue and one scheduler thread runs the steps.
 * This object owns the HTTP thread pool and the admission queue, never the step loop. Output
 * returns per request through a Sink (sink.h) that `Request::sink` points at.
 *
 * Backpressure is admission control, not buffering: a full queue answers 429 with Retry-After.
 * Accepting everything would turn a load spike into a later out-of-memory kill of every in-flight
 * request instead of refusing the marginal ones.
 *
 * Cancellation is the normal case for agent clients. A dropped connection, seen on the streaming
 * path as a failed write or a closed socket, becomes IScheduler::cancel, which frees the request's
 * KV blocks immediately.
 */
#pragma once
#include <functional>
#include "http.h"
#include "iface.h"
#include "liveview.h"
#include "metrics.h"
#include "oai.h"
#include "sink.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct Deps {
    IScheduler*       sched   = nullptr;   /* required */
    ITokenizer*       tok     = nullptr;   /* required */
    IChatTemplate*    chat    = nullptr;   /* absent -> /v1/chat/completions is 501 */
    IGrammarCompiler* grammar = nullptr;   /* absent -> response_format is 501 */
    IEmbedder*        embed   = nullptr;   /* absent -> /v1/embeddings is 501 */
    IMultimodal*      mm      = nullptr;   /* absent -> image and audio parts are 501 */
    IImageGen*        images  = nullptr;   /* absent -> the images endpoints are 501 */
    IAudioGen*        audio   = nullptr;   /* absent -> /v1/audio/speech is 501 */
    IVideoGen*        videos  = nullptr;   /* absent -> /v1/videos is 501 */
    const SpeechFormat* speech = nullptr;  /* absent -> /v1/audio/transcriptions is 501 */

    /* Engine-side state the server cannot see (per-card VRAM, the expert plane), as a snapshot
     * function since the server holds no device handle. The same function feeds the live view
     * (liveview.h), so the terminal and the web page agree. Absent, the dashboard shows only the
     * scheduler's metrics and omits the cards. */
    LiveView::Source  live;

    /* The control vector (core/cvec.h), as the server sees it: whether the engine found one for
     * this container, the runtime default for requests that name no choice, and the switch the
     * admin endpoint flips. Absent (the function members null) means the feature does not exist on
     * this deployment: a request that asks for it is refused by name, and the dashboard shows no
     * control. Nothing here reloads the model. */
    struct ControlVectorSwitch {
        std::function<bool()>        available;
        std::function<bool()>        get_default;
        std::function<void(bool)>    set_default;
        std::function<std::string()> json;
    };
    ControlVectorSwitch control_vector;
};

struct ServerOptions {
    std::string model_id = "radiance";
    /* An embedding model (rad_declare_pooled) generates nothing: /v1/chat/completions and
     * /v1/completions answer 501 naming /v1/embeddings. */
    bool        embedding_model = false;
    /* An image generator (rad_declare_generator) generates no text: the text endpoints answer
     * 501 naming /v1/images/generations and /v1/images/edits. */
    bool        image_model = false;
    /* An audio generator (rad_declare_audio_generator) generates no text either: the text
     * endpoints answer 501 naming /v1/audio/speech. */
    bool        audio_model = false;
    /* A video generator (rad_declare_video_generator) generates no text either: the text
     * endpoints answer 501 naming /v1/videos. */
    bool        video_model = false;
    /* How long a finished video's file is kept for /v1/videos/{id}/content, in seconds, and how
     * many finished videos are kept at most (the oldest go first). */
    int64_t     video_ttl_s = 3600;
    int64_t     max_videos = 32;
    /* The decision protocol from the container's `decision.type`: "d1" serves /v1/systemone
     * (systemone.h); empty answers it 501. Never inferred from a name. */
    std::string decision_type;
    /* What --model named, reported as the model entry's `root` as vLLM reports its loaded path
     * (the served name says nothing about the file behind it). Empty omits the field. */
    std::string model_path;
    std::string host = "0.0.0.0";
    int         port = 8000;
    int         n_threads = 0;

    /* Requests that may be admitted and unfinished at once; beyond it, 429. Defaults to 8 x
     * max_seqs: enough for a full batch plus a prefill queue, while bounding the memory waiting
     * requests hold. */
    int64_t     queue_depth = 0;
    int64_t     max_seqs = 256;
    int         retry_after_s = 1;

    int64_t     max_ctx = 0;
    int64_t     default_max_tokens = 0;     /* 0: what the context leaves (OaiLimits) */
    int64_t     max_tokens_cap = 0;         /* 0: none */
    int         max_n = 8;
    int64_t     max_stops = 64;
    int64_t     max_stop_bytes = 4096;

    bool        supports_logprobs = false;
    /* Deployment default for `reasoning_effort`; empty for the template's own. See
     * OaiLimits::default_reasoning_effort. */
    std::string default_reasoning_effort;
    /* Deployment default for `thinking_token_budget`, -1 for none. See
     * OaiLimits::default_thinking_budget. */
    int64_t     default_thinking_budget = -1;
    /* Sampling for a request that sends no sampler fields; only the sampler knobs are read. See
     * OaiLimits::default_sampling. */
    SamplingParams default_sampling;
    bool        default_dry_breakers_set = false;
    /* The deployment's chat template variables and reply format. See OaiLimits. */
    std::map<std::string, std::string> default_template_kwargs;
    /* --reasoning-effort-map. See OaiLimits::reasoning_effort_map. */
    std::map<std::string, std::string> reasoning_effort_map;
    std::string reasoning_format = "auto";
    bool        allow_image = false;
    bool        allow_video = false;
    bool        allow_audio = false;

    /* What /server_info and /version report about the engine. `engine_info` is a JSON object as
     * text, spliced in verbatim under "engine": the server knows nothing of the device layer,
     * container or placement plan, and an opaque blob need not grow with every new engine fact. */
    std::string version = RAD_VERSION;
    std::string engine_info;
    /* The declared compute graph as JSON, for the Model view. A provider called per request, not
     * a string rendered once: the dump includes `OpInfo::issued`, which a pre-first-step snapshot
     * would report as false for every op, hiding dead declarations.
     *
     * Regenerating reads the Program on the HTTP thread while the step path writes that field. It
     * is a monotonic false -> true bool, so at worst a mid-step fetch shows an op unissued one step
     * early. Registry::select is called with record = false and mutates nothing else. */
    std::function<std::string()> graph_json;
    /* The startup report as JSON (core/diag/diag.h): options in force, the machine as it is now
     * (the provider may reuse a reading for a second), and what on it costs speed. Without an API
     * key, loopback clients only (Server::handle_diagnostics). A provider because link speed and
     * clock only mean something under load: queried during a long prefill, it tells a link parked
     * below its maximum from one at rest. */
    std::function<std::string()> diagnostics_json;
    /* Runtime LoRA (Engine::lora_acquire): a request's `model` naming a loaded adapter gets that
     * adapter's slot, load uid and a hold on it. Unset with LoRA off, and then any `model` is
     * served by the base. `lora_names` lists the adapters a request may name (not the draining
     * ones). */
    std::function<int(const std::string&, int32_t*, uint64_t*, std::shared_ptr<void>*)> lora_acquire;
    std::function<std::vector<std::string>()>                  lora_names;
    /* POST /v1/load_lora_adapter and /v1/unload_lora_adapter (vLLM's names and bodies). An unload
     * of an adapter requests still use drains it: RAD_OK with *in_flight > 0 (Engine::lora_unload). */
    std::function<int(const std::string&, const std::string&, std::string*)> lora_load;
    std::function<int(const std::string&, std::string*, int*)>              lora_unload;
    /* A generator's adapters (Engine::gen_lora_acquire): the images and videos endpoints' `loras`
     * and <lora:...> tags get a slot holding that combination, a hold on it and a note on what was
     * left out; `gen_lora_names` is GET /v1/loras. Unset without adapters. */
    std::function<int(const std::vector<std::pair<std::string, double>>&, int32_t*,
                      std::shared_ptr<void>*, std::string*, std::string*)> gen_lora_acquire;
    std::function<std::vector<std::string>()>                                gen_lora_names;

    std::string api_key;
    bool        cors = true;

    /* The transport's timeouts and body bound. See TransportOptions. */
    int         read_timeout_s = 30;
    int         write_timeout_s = 600;
    int         keep_alive_timeout_s = 5;
    size_t      max_body_bytes = 512ull << 20;
};

class Server {
public:
    Server(const Deps& d, const ServerOptions& o);
    /* Stop the scheduler before destroying the Server: `Request::sink` is a raw pointer into
     * storage this object owns, and an unfinished request still writes into it. The destructor
     * waits a bounded time for cancelled requests, then leaks the rest rather than free memory a
     * running step may touch. */
    ~Server();

    /* Every endpoint. Tests drive this directly: same handlers, no socket. */
    const Router& router() const { return router_; }

    int  run();            /* binds, serves, blocks. RAD_OK or negative. */

    /* Renders the deployment's template defaults once at startup, since a default the template
     * refuses would 400 every request relying on it. A one-message chat is parsed as a request
     * would be: with no defaults, then with --chat-template-kwargs, then also --reasoning-effort.
     * A failure at the first stage is not about the defaults and is not reported. `kwargs_why` /
     * `effort_why` stay empty or hold the template's refusal at that stage. */
    void check_template_defaults(std::string* kwargs_why, std::string* effort_why) const;
    /* Checks the deployment's thinking_token_budget with a one-message chat, as a request would
     * send it. Empty when the model's reasoning block can take one; otherwise the 400 message a
     * request naming the budget would get. */
    std::string check_thinking_budget() const;
    /* The reasoning efforts the chat template takes, found by parsing a one-message chat with each
     * known word (low, medium, high, xhigh, max), with no deployment defaults. If the template
     * also takes a nonsense word it takes any word, and the result is empty. Used to refuse words
     * outside the set and for /server_info; called once, before run(). */
    std::vector<std::string> probe_reasoning_efforts();
    /* The effort the template applies to a request that names none, found by the probe above as
     * the one whose render is the same; "" when none matches or the probe found no set. */
    const std::string& template_default_effort() const { return template_default_effort_; }
    /* Renders each --reasoning-effort-map target as a request's effort. Empty when the template
     * takes all of them; otherwise the first refusal, naming the target. */
    std::string check_reasoning_effort_map() const;
    void stop();
    int  port() const { return transport_.port(); }   /* 0 until run() is accepting */

    int64_t  in_flight() const { return in_flight_.load(std::memory_order_relaxed); }
    /* The admission ceiling as resolved: ServerOptions::queue_depth, or 8 x max_seqs for 0. */
    int64_t  queue_cap() const { return queue_depth_; }

    /* Records how long bringup took, called by Engine::serve just before it starts listening
     * (the same moment the startup watch ends). `begin_ns` is rad_mono_ns from the top of
     * Engine::start(); 0 leaves the figure unset. run() blocks serving forever, so the value is
     * handed in rather than taken inside it. */
    void note_ready(uint64_t begin_ns);

private:
    std::string template_default_effort_;
    struct Generation;
    struct TranscriptionJob;   /* speech.cpp */
    struct RealtimeSession;    /* realtime.cpp */
    struct SpeechSession;      /* speech_stream.cpp */
    struct VideoJob;           /* videos.cpp */
    struct StreamState;

    static UsageOut usage_of(const OaiRequest& r, const std::vector<Generation>& gens);
    static TimingsOut timings_of(const Sink* const* sinks, size_t n, int64_t received_ns);
    static TimingsOut timings_of(const std::vector<Generation>& gens, int64_t received_ns);

    void register_routes();

    /* ------------------------------------------------------------------ server.cpp */
    HttpResponse handle_chat(const HttpRequest& req);
    HttpResponse handle_completion(const HttpRequest& req);
    HttpResponse handle_messages(const HttpRequest& req);
    HttpResponse handle_responses(const HttpRequest& req);
    HttpResponse handle_native_completion(const HttpRequest& req);
    HttpResponse handle_infill(const HttpRequest& req);
    HttpResponse handle_embeddings(const HttpRequest& req);
    HttpResponse handle_systemone(const HttpRequest& req);
    HttpResponse handle_apply_template(const HttpRequest& req);
    using ToChat = int (*)(const std::string& body, const OaiLimits& lim, std::string* chat,
                           std::string* extra, ApiError* err);
    HttpResponse count_tokens(const HttpRequest& req, ToChat to_chat, ApiShape shape,
                              const char* object);
    /* OpenAI's /v1/audio/transcriptions, from the model's own RadAudioFormat (speech.cpp). */
    HttpResponse handle_transcription(const HttpRequest& req);
    /* OpenAI's realtime transcription session over a WebSocket at /v1/realtime: audio streamed
     * in, the transcript streamed out while it is spoken (realtime.cpp). */
    void handle_realtime(const HttpRequest& req, WsConn& ws);
    /* The session's thread: realtime_loop, an exception turned into an error event and a close,
     * and the pass in flight cancelled however the loop ends. The loop returns true when the
     * server ended the session, false when the peer left. */
    void realtime_run(RealtimeSession& s, WsConn& ws);
    bool realtime_loop(RealtimeSession& s, WsConn& ws);
    HttpResponse handle_models(const HttpRequest& req);
    HttpResponse handle_loras(const HttpRequest& req);
    HttpResponse handle_metrics(const HttpRequest& req);
    std::optional<HttpResponse> refuse_text(const std::string& peer,
                                            ApiShape shape = ApiShape::OpenAI);
    /* A parsed generation request, in any API's shape: admitted, run, answered in `r.shape`. */
    HttpResponse serve(const HttpRequest& req, OaiRequest& r);
    json native_result(const OaiRequest& r, const Generation& g, const std::string& content);

    /* ------------------------------------------------------------------ images.cpp */
    HttpResponse handle_images(const HttpRequest& req, bool edit);

    /* ------------------------------------------------------------------ audio_gen.cpp */
    HttpResponse handle_speech(const HttpRequest& req);
    /* The audio a speech request describes, `text` its input, as the generator's request; -1 with
     * `err` naming the field. Shared by the route and the realtime session. */
    int speech_request(const json& body, const std::string& text,
                       std::unique_ptr<gen::AudioRequest>* out, ApiError* err);
    HttpResponse stream_speech(std::shared_ptr<gen::AudioTicket> ticket, const AudioGenInfo& info,
                               int64_t n_tok, uint64_t seed, bool sse, bool wav);

    /* ------------------------------------------------------------------ videos.cpp
     * OpenAI's video jobs: created by POST, made in the background, polled and downloaded. */
    HttpResponse handle_video_create(const HttpRequest& req);
    HttpResponse handle_video_list(const HttpRequest& req);
    HttpResponse handle_video_get(const HttpRequest& req);
    HttpResponse handle_video_delete(const HttpRequest& req);
    HttpResponse handle_video_content(const HttpRequest& req);
    /* A job's thread: the ticket's events into progress and an MP4 file. */
    void video_work(std::shared_ptr<VideoJob> j);
    /* Drops expired and surplus finished jobs; under videos_m_. */
    void sweep_videos_locked(int64_t now);
    /* Cancels every job and joins its thread; at stop(). */
    void stop_videos();
    static json video_json(const VideoJob& j);

    /* ------------------------------------------------------------------ speech_stream.cpp */
    /* vLLM-Omni's realtime speech session over a WebSocket at /v1/audio/speech/stream: text
     * streamed in, each sentence's audio streamed out as soon as the sentence is complete. */
    void handle_speech_stream(const HttpRequest& req, WsConn& ws);
    void speech_stream_run(SpeechSession& s, WsConn& ws);
    bool speech_stream_loop(SpeechSession& s, WsConn& ws);

    /* ------------------------------------------------------------------ admin.cpp
     * vLLM's non-generation surface, answered on the HTTP thread without the step path, the
     * admission queue or a Sink. */
    HttpResponse handle_health(const HttpRequest& req);
    HttpResponse handle_ping(const HttpRequest& req);
    HttpResponse handle_version(const HttpRequest& req);
    HttpResponse handle_props(const HttpRequest& req);
    HttpResponse handle_load(const HttpRequest& req);
    HttpResponse handle_tokenize(const HttpRequest& req);
    HttpResponse handle_detokenize(const HttpRequest& req);
    HttpResponse handle_tokenizer_info(const HttpRequest& req);
    HttpResponse handle_server_info(const HttpRequest& req);
    HttpResponse handle_diagnostics(const HttpRequest& req);
    HttpResponse handle_lora_admin(const HttpRequest& req, bool load);
    /* What this deployment can do (chat, tools, reasoning, grammar, media, logprobs, reasoning
     * efforts), shared by /server_info's `capabilities` and the model entry so they agree. */
    json capabilities() const;
    HttpResponse handle_reset_prefix_cache(const HttpRequest& req);
    /* The draft calibration's phase boundary (--draft-profile-out): 501 outside a calibration, 409
     * when the engine was not quiet, 200 with the epoch it opened. */
    HttpResponse handle_draft_calib_boundary(const HttpRequest& req);

    /* ------------------------------------------------------------------ dashboard.cpp */
    HttpResponse handle_dashboard(const HttpRequest& req);
    HttpResponse handle_stats(const HttpRequest& req);
    HttpResponse handle_graph(const HttpRequest& req);
    HttpResponse handle_sessions(const HttpRequest& req);
    /* The control vector's runtime switch (dashboard.cpp): GET reports the state, POST
     * {"enabled": bool} sets the default for requests that name no choice. Guarded by the API key
     * like every other route but the health probes and the page. */
    HttpResponse handle_control_vector(const HttpRequest& req);

    HttpResponse error_response(const ApiError& e, const std::string& peer,
                                ApiShape shape = ApiShape::OpenAI);
    /* Request log: one info line at admission and one from record() at the end (who asked what,
     * how long, how it ended). The debug level adds the opening of the last user message. */
    void log_admitted(const OaiRequest& r) const;

    /* Admission: reserves `n` places in the queue, or returns the 429 to answer when it is full.
     * More places than the queue has can never be admitted, so that is a 400 under `field` (the
     * request field that multiplied them) naming the bound, not a 429 retried forever. Sweeps the
     * graveyard first. */
    std::optional<HttpResponse> admit(int64_t n, const std::string& peer,
                                      ApiShape shape = ApiShape::OpenAI, const char* field = "n");
    void  release(int64_t n);
    void  sweep_abandoned();
    /* Wait, bounded, for the scheduler to finish everything it was told to cancel. */
    void  drain_graveyard(int timeout_ms);

    /* Turn a parsed request into `n_choices` submitted engine requests. */
    int   submit_all(const OaiRequest& r, std::vector<Generation>* gens, ApiError* err);
    void  cancel_all(std::vector<Generation>& gens, bool count_disconnect);
    /* While `gens` run: once the client has left (499) or the server is stopping, every
     * generation is cancelled and retired, `n` queue places are released, and the answer is
     * returned. Nothing otherwise. A generation request (`r`) is recorded as an abort, as a
     * dropped stream is. */
    std::optional<HttpResponse> abandoned(const HttpRequest& req, std::vector<Generation>& gens,
                                          int64_t n, ApiShape shape = ApiShape::OpenAI,
                                          const OaiRequest* r = nullptr);
    void  retire(std::vector<Generation>& gens);

    /* Drain sinks into decoded text. Returns true if anything new arrived. */
    bool  pump(std::vector<Generation>& gens, const OaiRequest& r);
    bool  all_done(const std::vector<Generation>& gens) const;

    HttpResponse blocking_completion(const HttpRequest& req, OaiRequest& r);
    HttpResponse streaming_completion(const HttpRequest& req, OaiRequest& r);

    /* `quiet` says the request's line at the debug level: one of the many small passes a
     * realtime session makes, which its own lines account for. */
    void  record(const OaiRequest& r, const std::vector<Generation>& gens, Finish overall,
                 bool quiet = false);

    Deps           deps_;
    ServerOptions  opt_;
    OaiLimits      lim_;
    OaiDeps        odeps_;
    Router         router_;
    Metrics        metrics_;
    HttpTransport  transport_;

    std::atomic<int64_t> in_flight_{0};
    int64_t              queue_depth_ = 0;
    std::atomic<int64_t> realtime_sessions_{0};
    std::atomic<int64_t> speech_sessions_{0};
    std::atomic<bool>    stopping_{false};
    int64_t              started_at_ = 0;
    /* Seconds from the top of Engine::start() to the moment the server began listening; 0 until
     * note_ready(). */
    int64_t              startup_s_ = 0;

    /* Sinks whose HTTP side is gone but whose engine side may still be mid-step. Held until the
     * scheduler reports the request terminal, because Request::sink is a raw pointer and freeing
     * it under a running step is a use-after-free. */
    std::mutex                          grave_m_;
    std::vector<std::shared_ptr<Sink>>  grave_;

    /* The video jobs, oldest first, and the threads that make them (videos.cpp); deleted jobs
     * wait in `videos_gone_` until their threads are done. */
    std::mutex                              videos_m_;
    std::vector<std::shared_ptr<VideoJob>>  videos_;
    std::vector<std::shared_ptr<VideoJob>>  videos_gone_;
};

}  /* namespace server */
}  /* namespace rad */
