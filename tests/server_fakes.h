/* tests/server_fakes.h -- the engine behind the HTTP layer, faked, for the server's tests.
 *
 * Router::dispatch is the call the transport makes, so a test drives every handler exactly as for
 * a real request; only what the server reaches through core/server/iface.h is faked here. One
 * token per byte throughout, so an expected completion is a literal and a stop string is a
 * substring test.
 *
 * FakeScheduler runs in one of two ways. Threaded (the default): each submitted request gets a
 * thread that writes `reply` into its sink. Manual: submit only records the request, and the test
 * writes tokens into the sink from its own thread, so it decides exactly what each drain of a
 * stream sees. Either way it checks the reap contract (IScheduler::reap): an id is reaped at most
 * once, only after its sink finished, and only if this scheduler issued it.
 */
#pragma once

#include "rad_test.h"

#include "gen/audio.h"
#include "gen/gen.h"
#include "mm/processor.h"
#include "server/adapters.h"
#include "server/http.h"
#include "server/json_partial.h"
#include "server/oai.h"
#include "server/server.h"
#include "server/sink.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace rad;
using namespace rad::server;
using json = nlohmann::ordered_json;

/* ================================================================== text */

/* One token per byte. Trivially round-trips, so a stop string is a substring test and an expected
 * completion is a literal rather than a vocabulary lookup. */
struct FakeDetok : IDetokenizer {
    std::string push(int32_t t) override {
        if (t <= 0) return "";
        return std::string(1, (char)t);
    }
    std::string flush() override { return ""; }
};

struct FakeTokenizer : ITokenizer {
    mutable bool last_parse_special = false;
    /* A vocabulary with llama's three fill-in-the-middle markers as special tokens, for /infill.
     * With `repo` also the repository-level pair, `<REPO>` and `<filename>`. */
    bool fim = false;
    bool repo = false;
    /* What vocab_info() reports about BOS: whether the tokenizer adds one, and its id. */
    bool add_bos = false;
    static const char* fim_name(int32_t id) {
        return id == 1001 ? "<PRE>" : id == 1002 ? "<SUF>" : id == 1003 ? "<MID>"
             : id == 1004 ? "<REPO>" : id == 1005 ? "<filename>" : nullptr;
    }
    bool special(int32_t id) const {
        return (fim && id >= 1001 && id <= 1003) || (repo && (id == 1004 || id == 1005));
    }
    std::vector<int32_t> encode(const std::string& s, bool add_special,
                                bool parse_special) const override {
        last_parse_special = parse_special;
        if (parse_special)
            for (int32_t id = 1001; id <= 1005; ++id)
                if (special(id) && s == fim_name(id)) return { id };
        std::vector<int32_t> v;
        v.reserve(s.size() + 1);
        if (add_bos && add_special) v.push_back(bos());
        for (unsigned char c : s) v.push_back((int32_t)c);
        return v;
    }
    std::string decode(const std::vector<int32_t>& t) const override {
        std::string s;
        for (int32_t x : t) if (x > 0) s += token_bytes(x);
        return s;
    }
    std::string token_bytes(int32_t id) const override {
        if (special(id)) return fim_name(id);
        return std::string(1, (char)id);
    }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return 1; }
    VocabInfo vocab_info() const override {
        VocabInfo i;
        i.n_tokens = 1006;
        i.bos = 1;
        i.eos = 0;
        i.add_bos = add_bos;
        return i;
    }
    /* Stop strings are deliberately not given to the fake decoder, so these tests exercise the
     * server's own scan. The real Detokenizer can take them and report stopped(). */
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                              const std::vector<std::string>&) const override {
        return std::unique_ptr<IDetokenizer>(new FakeDetok());
    }
};

struct FakeChatTemplate : IChatTemplate {
    bool tools_ok = true;
    /* What the last render was asked for, so a test can assert a request's options reached the
     * template. */
    mutable ChatRenderOptions last_opt;
    mutable int last_n_tools = 0;
    /* The last render's prompt, and how many tool messages it was given. */
    mutable std::string last_prompt;
    mutable int last_tool_msgs = 0;

    int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
              ChatRender* out, std::string* why) const override {
        last_opt = opt;
        last_n_tools = (int)tools.size();
        /* A template that raises on one variable, as Qwen3.8's does on an effort it does not
         * know. */
        if (!refuse_kwarg.empty() && opt.template_kwargs.count(refuse_kwarg)) {
            if (why) *why = "the template does not take " + refuse_kwarg;
            return RAD_E_UNSUPPORTED;
        }
        /* And one with a set of its own, as Qwen3.8's xhigh|medium|low. */
        if (!efforts.empty() && opt.template_kwargs.count("reasoning_effort")) {
            const std::string& v = opt.template_kwargs.at("reasoning_effort");
            if (std::find(efforts.begin(), efforts.end(), v) == efforts.end()) {
                if (why) *why = "Jinja Exception: Unexpected reasoning effort " + v;
                return RAD_E_UNSUPPORTED;
            }
        }
        std::string s;
        if (!tools.empty()) s += "[tools:" + std::to_string(tools.size()) + "]";
        if (!opt.enable_thinking) s += "[nothink]";
        for (const auto& kv : opt.template_kwargs) s += "[" + kv.first + "=" + kv.second + "]";
        if (!default_effort.empty() && !opt.template_kwargs.count("reasoning_effort"))
            s += "[reasoning_effort=" + default_effort + "]";
        size_t n_msg = 0;
        int tool_msgs = 0;
        for (const auto& m : messages) {
            /* A template that leaves out every message after the first (some drop a system turn
             * or an old picture). */
            if (drop_after_first && n_msg++ > 0) continue;
            const std::string role = m.value("role", std::string("?"));
            s += "<" + role + ">";
            if (role == "tool") ++tool_msgs;
            /* An assistant turn of tool calls only has no content. */
            const auto fc = m.find("content");
            if (fc == m.end()) continue;
            const auto& c = *fc;
            if (c.is_string()) s += c.get<std::string>();
            else for (const auto& p : c) {
                const std::string type = p.value("type", std::string());
                /* Text parts and the media markers lower_content_parts leaves in their place, both
                 * joined with nothing, as the chat layer joins a marker. */
                if (type == "text" || type == "media_marker") s += p.value("text", std::string());
            }
        }
        if (opt.add_generation_prompt) s += "<assistant>";
        out->prompt = s;
        last_prompt = s;
        last_tool_msgs = tool_msgs;
        /* A tool-calling template's grammar is lazy, and the laziness must survive the trip: a
         * non-lazy one would constrain the whole reply to be a call. */
        if (!tools.empty()) {
            out->grammar = "root ::= \"TOOL:\" [^\\n]*";
            out->grammar_lazy = true;
            out->grammar_triggers.push_back({ /*type=word*/ 1, "TOOL:" });
        } else {
            /* A schema becomes a grammar, as the real template builds one from it. */
            out->grammar = !opt.grammar.empty() || opt.json_schema.empty() ? opt.grammar
                                                                           : "root ::= object";
        }
        /* The parser is built per render and reads calls only for a request that sent tools, the
         * contract the real bridge keeps (IReplyParser). */
        if (real_parser) out->parser = real_parser;
        else if (with_parser && think_reply) out->parser = std::make_shared<FakeThinkParser>();
        else if (with_parser) out->parser = std::make_shared<FakeReplyParser>(!tools.empty());
        out->think_open = think_open;
        out->think_close = think_close;
        out->think_starts_open = think_starts_open;
        out->additional_stops = additional_stops;
        return RAD_OK;
    }
    bool supports_tools() const override { return tools_ok; }
    bool supports_reasoning() const override { return false; }
    std::string tool_refusal() const override { return refusal; }

    bool        with_parser = false;
    /* The reply reads as reasoning then content, not as a tool call. */
    bool        think_reply = false;
    /* A real reply parser to hand out instead of the fake one. */
    std::shared_ptr<IReplyParser> real_parser;
    std::string refusal;
    std::string refuse_kwarg;
    std::vector<std::string> efforts;   /* JSON text, as the template context carries them */
    /* The effort a render naming none is written as, as Qwen3.8's template writes xhigh. */
    std::string default_effort;
    /* The reasoning block a render reports. FakeTokenizer makes a one-byte marker one token. */
    std::string think_open, think_close;
    bool        think_starts_open = false;
    /* The format's end-of-turn markers, which join the caller's stop strings. */
    std::vector<std::string> additional_stops;
    bool        drop_after_first = false;

    /* Reads "TOOL:<name>:<json...>": everything after the second colon is the arguments, however
     * incomplete, arriving a piece at a time, the shape the streaming path must handle. With calls
     * off every byte is content. The call is open when the reply ends before its arguments are
     * whole, and its text is then the reply. */
    struct FakeReplyReader : IReplyReader {
        explicit FakeReplyReader(bool calls) : calls_(calls) {}
        void feed(std::string_view t, std::vector<ReplyDelta>* out) override {
            text_.append(t);
            step(out, false);
        }
        void finish(std::vector<ReplyDelta>* out) override {
            step(out, true);
            msg_.open_call = !msg_.tool_calls.empty() && !json_complete(msg_.tool_calls[0].arguments);
        }
        ParsedMessage message() const override { return msg_; }
        std::string content_with_open_call(std::string_view reply) const override {
            return msg_.open_call ? std::string(reply) : msg_.content;
        }
    private:
        void emit(std::vector<ReplyDelta>* out, ReplyDelta::Kind k, std::string s) {
            if (out) out->push_back({ k, 0, std::move(s) });
        }
        void step(std::vector<ReplyDelta>* out, bool final) {
            static const std::string tag = "TOOL:";
            if (mode_ == 0) {
                /* Undecided while what has arrived could still be the start of the tag. */
                if (calls_ && !final && text_.size() < tag.size() &&
                    tag.compare(0, text_.size(), text_) == 0)
                    return;
                mode_ = calls_ && text_.rfind(tag, 0) == 0 ? 2 : 1;
            }
            if (mode_ == 1) {
                if (text_.size() > sent_) emit(out, ReplyDelta::Content, text_.substr(sent_));
                msg_.content = text_;
                sent_ = text_.size();
                return;
            }
            if (!named_) {
                const size_t c2 = text_.find(':', tag.size());
                if (c2 == std::string::npos) return;
                named_ = true;
                msg_.tool_calls.push_back({ "", text_.substr(tag.size(), c2 - tag.size()), "" });
                emit(out, ReplyDelta::CallBegin, msg_.tool_calls[0].name);
                sent_ = c2 + 1;
            }
            if (text_.size() > sent_) emit(out, ReplyDelta::CallArgs, text_.substr(sent_));
            msg_.tool_calls[0].arguments += text_.substr(sent_);
            sent_ = text_.size();
        }
        bool          calls_;
        int           mode_ = 0;
        bool          named_ = false;
        std::string   text_;
        size_t        sent_ = 0;
        ParsedMessage msg_;
    };

    struct FakeReplyParser : IReplyParser {
        explicit FakeReplyParser(bool calls) : calls_(calls) {}
        std::unique_ptr<IReplyReader> open() const override {
            return std::unique_ptr<IReplyReader>(new FakeReplyReader(calls_));
        }
        bool calls_;
    };

    /* A reply whose first line is reasoning and the rest is content: everything up to the first
     * newline arrives as Reasoning deltas, the rest as Content. The newline is the last byte
     * held back until it is confirmed, as the real parser holds back a possible marker. */
    struct FakeThinkReader : IReplyReader {
        void feed(std::string_view t, std::vector<ReplyDelta>* out) override {
            text_.append(t);
            step(out);
        }
        void finish(std::vector<ReplyDelta>* out) override {
            if (!resolved_ && text_.size() > reason_sent_) {
                emit(out, ReplyDelta::Reasoning, text_.substr(reason_sent_));
                reason_sent_ = text_.size();
                msg_.reasoning = text_;
                resolved_ = true;
            }
        }
        ParsedMessage message() const override { return msg_; }
        std::string content_with_open_call(std::string_view) const override {
            /* The reader never opens a call, so there is nothing to read back. */
            return msg_.content;
        }
    private:
        void emit(std::vector<ReplyDelta>* out, ReplyDelta::Kind k, std::string s) {
            if (out && !s.empty()) out->push_back({ k, 0, std::move(s) });
        }
        void step(std::vector<ReplyDelta>* out) {
            if (!resolved_) {
                const size_t nl = text_.find('\n');
                if (nl == std::string::npos) {
                    /* The last byte may still be the newline; hold it. */
                    const size_t up = text_.empty() ? 0 : text_.size() - 1;
                    if (up > reason_sent_) {
                        emit(out, ReplyDelta::Reasoning, text_.substr(reason_sent_, up - reason_sent_));
                        reason_sent_ = up;
                    }
                    return;
                }
                emit(out, ReplyDelta::Reasoning, text_.substr(reason_sent_, nl - reason_sent_));
                msg_.reasoning = text_.substr(0, nl);
                reason_sent_ = nl;
                nl_ = cont_sent_ = nl + 1;
                resolved_ = true;
            }
            if (text_.size() > cont_sent_) {
                emit(out, ReplyDelta::Content, text_.substr(cont_sent_));
                msg_.content = text_.substr(nl_);
                cont_sent_ = text_.size();
            }
        }
        std::string text_;
        ParsedMessage msg_;
        bool        resolved_ = false;
        size_t      reason_sent_ = 0;
        size_t      cont_sent_ = 0;
        size_t      nl_ = 0;
    };

    struct FakeThinkParser : IReplyParser {
        std::unique_ptr<IReplyReader> open() const override {
            return std::unique_ptr<IReplyReader>(new FakeThinkReader());
        }
    };
};

/* Qwen's reply format, read by the real parser (core/text/chatparse.h): a reply starting inside
 * <think>, XML-ish calls. `tools` must outlive the parser. */
inline std::shared_ptr<IReplyParser> qwen_reply_parser(const json& tools) {
    ChatFormat f;
    f.reasoning_open = "<think>\n";
    f.reasoning_close = "\n</think>\n\n";
    f.reasoning_breaks = { "<tool_call>\n" };
    f.call_open = "<tool_call>\n";
    f.call_close = "</tool_call>";
    f.name_prefix = "<function=";
    f.name_suffix = ">\n";
    f.call_end = "</function>\n";
    f.arg_prefix = "<parameter=";
    f.arg_name_suffix = ">\n";
    f.arg_value_suffix = "</parameter>\n";
    f.triggers = { "<tool_call>\n" };
    std::shared_ptr<const ReplyFormat> rf;
    std::string why;
    if (ReplyFormat::compile(f, &rf, &why) < 0) return nullptr;
    ReplyOptions ro;
    ro.generation_prompt = "<|im_start|>assistant\n<think>\n";
    ro.tools = &tools;
    std::shared_ptr<const ReplyParser> rp;
    if (ReplyParser::create(rf, ro, &rp, &why) < 0) return nullptr;
    return std::make_shared<ReplyParserBridge>(rp);
}

struct FakeGrammar : IGrammarCompiler {
    int from_json_schema(const json&, std::string& out, std::string&) const override {
        out = "root ::= object";
        return RAD_OK;
    }
    /* "bad" is the one grammar this fake refuses, so a test can drive the admission-time
     * rejection without a real GBNF parser. */
    int compile(const std::string& g, std::shared_ptr<const GbnfProgram>*,
                std::string& err) const override {
        if (g.find("bad") != std::string::npos) { err = "left recursion"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

/* Takes any media and says it fills `expand`, by default one position, so a request's media
 * parts reach the server's own validation. Records the bytes and kind of every part it was
 * handed, in order. */
struct FakeMultimodal : IMultimodal {
    std::vector<int32_t> expand = { 0 };
    bool accept_image = true, accept_video = true, accept_audio = true;
    std::vector<std::pair<MediaKind, std::vector<uint8_t>>> got;
    bool accepts(MediaKind k) const override {
        return k == MediaKind::Image ? accept_image : k == MediaKind::Video ? accept_video
                                                                            : accept_audio;
    }
    int prepare(MediaKind k, const std::vector<uint8_t>& bytes, const MediaOpts&,
                std::shared_ptr<mm::Item>* out, std::string*) override {
        got.emplace_back(k, bytes);
        auto it = std::make_shared<mm::Item>();
        it->expand = expand;
        *out = std::move(it);
        return RAD_OK;
    }
};

/* An embedding model's vector width, input bound and named prompts. */
/* The audio side: the "file" is its own samples, one byte a sample -- 0 silence, anything else a
 * tone -- and every clip handed to prepare_audio is recorded. */
struct FakeAudio : IMultimodal {
    int64_t max_clip = 16000;
    /* Nonzero: a clip's placeholder run is one token for this many samples, so a fake model can
     * tell how much it was given to hear. */
    int64_t per_token = 0;
    /* Throw from prepare_audio: a fault the route's thread did not expect. */
    bool throw_on_prepare = false;
    std::mutex mu;
    std::vector<int64_t> clips;
    bool accepts(MediaKind k) const override { return k == MediaKind::Audio; }
    int prepare(MediaKind, const std::vector<uint8_t>&, const MediaOpts&,
                std::shared_ptr<mm::Item>*, std::string* why) override {
        if (why) *why = "not this way";
        return RAD_E_UNSUPPORTED;
    }
    int32_t audio_rate() const override { return 16000; }
    int64_t audio_max_samples() const override { return max_clip; }
    int decode_audio(const std::vector<uint8_t>& bytes, int64_t max_samples,
                     std::vector<float>* out, std::string* why) override {
        out->clear();
        for (size_t i = 0; i < bytes.size(); ++i) out->push_back(bytes[i] ? (i & 1 ? 0.5f : -0.5f) : 0.0f);
        if ((int64_t)out->size() > max_samples) { if (why) *why = "too long"; return RAD_E_INVAL; }
        return RAD_OK;
    }
    int prepare_audio(const std::vector<float>& samples, std::shared_ptr<mm::Item>* out,
                      std::string*) override {
        if (throw_on_prepare) throw std::runtime_error("the fake front end threw");
        {
            std::lock_guard<std::mutex> lk(mu);
            clips.push_back((int64_t)samples.size());
        }
        auto it = std::make_shared<mm::Item>();
        it->expand.assign(per_token > 0 ? (size_t)std::max<int64_t>(1, (int64_t)samples.size() /
                                                                         per_token)
                                        : 3,
                          7);
        *out = std::move(it);
        return RAD_OK;
    }
};

/* An audio "file" of n tone samples, in FakeAudio's encoding. */
inline std::string tone(size_t n) { return std::string(n, '\x01'); }

/* ---- the realtime session's client side */

/* 16-bit PCM: a 440 Hz tone at half scale, or silence, for `ms` milliseconds at `rate`. */
inline std::string pcm16(int32_t rate, int ms, bool tone, int64_t* phase = nullptr) {
    const int64_t n = (int64_t)rate * ms / 1000;
    int64_t at = phase ? *phase : 0;
    std::string b;
    for (int64_t i = 0; i < n; ++i, ++at) {
        const double v = tone ? 0.5 * std::sin(2.0 * M_PI * 440.0 * (double)at / rate) : 0.0;
        const int16_t x = (int16_t)std::lround(v * 32767.0);
        b.push_back((char)(x & 0xff));
        b.push_back((char)((x >> 8) & 0xff));
    }
    if (phase) *phase = at;
    return b;
}

inline json append_event(const std::string& pcm) {
    json j;
    j["type"] = "input_audio_buffer.append";
    j["audio"] = base64_encode(pcm.data(), pcm.size());
    return j;
}

inline json session_update(int rate, const json& turn_detection = json(),
                           const std::string& language = "") {
    json j;
    j["type"] = "session.update";
    j["session"]["type"] = "transcription";
    j["session"]["audio"]["input"]["format"]["type"] = "audio/pcm";
    j["session"]["audio"]["input"]["format"]["rate"] = rate;
    j["session"]["audio"]["input"]["turn_detection"] = turn_detection;
    if (!language.empty()) j["session"]["audio"]["input"]["transcription"]["language"] = language;
    return j;
}

/* A model that hears a word every half second, held to what it is told it already said: its reply
 * is "lang English<t>" and the words so far with a full stop -- less whatever of that the prompt
 * already ends with, after its last '|', the way a model continues a reply it is handed the start
 * of. One placeholder is a tenth of a second (FakeAudio::per_token). */
inline const char* const kWords[] = { "one", "two", "three", "four", "five", "six", "seven",
                                      "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                                      "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                      "nineteen", "twenty" };
inline std::string heard_words(size_t n) {
    std::string t;
    for (size_t i = 0; i < n && i < 20; ++i) t += (i ? " " : "") + std::string(kWords[i]);
    return t + ".";
}
inline std::string asr_reply(const std::vector<int32_t>& prompt) {
    size_t tenths = 0, bar = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (prompt[i] == 7) ++tenths;
        if (prompt[i] == '|') bar = i + 1;
    }
    std::string tail;
    for (size_t i = bar; i < prompt.size(); ++i) tail.push_back((char)prompt[i]);
    const std::string full = "lang English<t> " + heard_words(tenths / 5);
    return full.compare(0, tail.size(), tail) == 0 ? full.substr(tail.size()) : full;
}

struct FakeEmbedder final : IEmbedder {
    IChatTemplate* tpl = nullptr;
    int64_t width = 768, longest = 8192;
    int64_t dim() const override { return width; }
    int64_t max_input() const override { return longest; }
    bool prompt(const std::string& name, std::string* text) const override {
        if (name != "query") return false;
        *text = "task: search result | query: ";
        return true;
    }
    IChatTemplate* message_template() const override { return tpl; }
};

/* An image generator that finishes every request on submit: `n` solid pictures, each preceded by
 * the previews it asked for, then the ticket finished with `finish_status`. A picture's pixel
 * holds its index and preview number so a test can tell them apart. Every request and reference
 * frame it was handed is kept. */
struct FakeImageGen : IImageGen {
    ImageGenInfo inf;
    int         submit_status = RAD_OK;
    std::string submit_why;
    int         finish_status = RAD_OK;
    std::string finish_why;
    std::vector<int32_t> picture_expand = { 7, 7, 7 };
    std::vector<std::unique_ptr<gen::Request>> got;
    std::vector<std::pair<int32_t, int32_t>> frames;   /* each reference's size as resized */

    FakeImageGen() {
        inf.name = "fake-image";
        inf.tmpl_text = "<s>{prompt}";
        inf.tmpl_edit = "<s>{images}|{prompt}";
        inf.tmpl_image = "[{n}:{image}]";
        inf.tmpl_image_sep = ",";
        inf.drop_prefix = "<s>";
        inf.latent_scale = 16;
        inf.size_multiple = 32;
        inf.max_tokens = 4096;           /* 1024 x 1024 pixels */
        inf.max_refs = 2;
        inf.ref_area = 64 * 64;
        inf.default_width = 256;
        inf.default_height = 128;
        inf.default_steps = 20;
        inf.max_lanes = 4;
        inf.max_cond_tokens = 4096;
    }
    const ImageGenInfo& info() const override { return inf; }
    int submit(std::unique_ptr<gen::Request> r, std::string* why) override {
        if (submit_status < 0) { *why = submit_why; return submit_status; }
        gen::Ticket& t = *r->ticket;
        t.in_tokens = (int64_t)r->pos.tokens.size() + (r->has_neg ? (int64_t)r->neg.tokens.size() : 0);
        t.out_tokens = (int64_t)r->n * (r->width / 16) * (r->height / 16);
        for (int32_t i = 0; i < r->n; ++i) {
            for (int32_t p = 0; p <= r->previews; ++p) {
                gen::Image im;
                im.index = i;
                im.w = r->width;
                im.h = r->height;
                im.preview = p < r->previews;
                im.preview_no = p;
                im.seed = r->seed + (uint64_t)i;
                im.px.assign((size_t)im.w * im.h * 4, 255);
                for (size_t k = 0; k < im.px.size(); k += 4) {
                    im.px[k] = (uint8_t)(40 * i);
                    im.px[k + 1] = (uint8_t)(60 * p);
                }
                t.push(std::move(im));
            }
        }
        t.finish(finish_status, finish_why);
        got.push_back(std::move(r));
        return RAD_OK;
    }
    int picture(const mm::Frame& f, std::shared_ptr<mm::Item>* out, std::string*) override {
        frames.emplace_back(f.w, f.h);
        auto it = std::make_shared<mm::Item>();
        it->expand = picture_expand;
        *out = std::move(it);
        return RAD_OK;
    }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
    /* The second tokenizer: one id a byte, then 1 (an end token). */
    int encode_aux(const std::string& text, std::vector<int32_t>* out) const override {
        if (!inf.aux_tokens) return RAD_E_UNSUPPORTED;
        out->clear();
        for (unsigned char c : text) out->push_back(100 + c);
        out->push_back(1);
        return RAD_OK;
    }
};

/* An audio generator that finishes every request on submit: `windows` windows of `per_window`
 * stereo samples, pushed in `order` (the driver may finish windows out of order), sample j of
 * window k at sample(k, j). Text containing "REFUSE" is refused by prepare(). With `hold` a
 * request is taken and left running, its ticket neither fed nor ended. A realtime session submits
 * from its own thread: read what was submitted through n_got() and request(). */
struct FakeAudioGen : IAudioGen {
    AudioGenInfo inf;
    int         submit_status = RAD_OK;
    std::string submit_why;
    int         finish_status = RAD_OK;
    std::string finish_why;
    std::vector<int32_t> order = { 1, 0, 2 };
    int32_t     per_window = 6;
    bool        hold = false;
    std::vector<std::unique_ptr<gen::AudioRequest>> got;
    mutable std::mutex got_m;

    FakeAudioGen() {
        inf.name = "fake-audio";
        inf.tmpl = "<s>{instructions}|{input}";
        inf.max_prompt = 64;
        inf.max_frames = 100;
        inf.frame_rate_num = 25;
        inf.frame_rate_den = 1;
        inf.sample_rate = 8000;
        inf.channels = 2;
        inf.default_steps = 30;
        inf.default_cfg = 1.7;
        inf.max_lanes = 4;
    }
    /* One sample past full scale (window 2's first), to show the clip. */
    static float sample(int32_t k, int32_t j) {
        return j == 0 && k == 2 ? 1.5f : (float)(k * 8 + j) / 37.0f - 0.4f;
    }
    const AudioGenInfo& info() const override { return inf; }
    int prepare(int, const std::string& in, std::string* out, std::string* why) const override {
        if (in.find("REFUSE") != std::string::npos) {
            *why = "the text was refused";
            return RAD_E_INVAL;
        }
        *out = in;
        return RAD_OK;
    }
    size_t n_got() const {
        std::lock_guard<std::mutex> lk(got_m);
        return got.size();
    }
    /* Request i's prompt as text (the fake tokenizer's ids are bytes) and its ticket. */
    std::string prompt_of(size_t i) const {
        std::lock_guard<std::mutex> lk(got_m);
        std::string t;
        for (int32_t c : got[i]->tokens) t += (char)c;
        return t;
    }
    std::shared_ptr<gen::AudioTicket> ticket_of(size_t i) const {
        std::lock_guard<std::mutex> lk(got_m);
        return got[i]->ticket;
    }
    int submit(std::unique_ptr<gen::AudioRequest> r, std::string* why) override {
        std::lock_guard<std::mutex> lk(got_m);
        if (submit_status < 0) { *why = submit_why; return submit_status; }
        if (hold) {
            got.push_back(std::move(r));
            return RAD_OK;
        }
        gen::AudioTicket& t = *r->ticket;
        t.in_tokens = (int64_t)r->tokens.size();
        t.frames = r->max_frames;
        t.windows = (int64_t)order.size();
        if (finish_status == RAD_OK)
            for (int32_t k : order) {
                gen::AudioChunk c;
                c.window = k;
                for (int32_t j = 0; j < per_window; ++j) c.samples.push_back(sample(k, j));
                t.push(std::move(c));
            }
        t.finish(finish_status, finish_why);
        got.push_back(std::move(r));
        return RAD_OK;
    }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
};

/* ================================================================== the scheduler */

struct FakeScheduler : IScheduler {
    /* Threaded mode: what each request's thread writes, one byte a token. In `hang` mode it stops
     * after three bytes and waits to be cancelled: a generation that never finishes on its own. */
    std::string reply = "hello";
    /* When set, the reply to each request, from its prompt: a model whose answer depends on what
     * it is asked. */
    std::function<std::string(const std::vector<int32_t>&)> reply_for;
    bool hang = false;
    /* Nonzero: after the reply, the request is failed with this status and reason, as the engine
     * fails every request it holds when a step fails. */
    int fail_status = 0;
    const char* fail_why = nullptr;
    /* Nonzero: every token after the first is a verify step that proposed this many drafts and
     * accepted one, as a speculating engine reports them. */
    int32_t drafts = 0;
    /* An embedding request's vector; empty gives none, as an engine that pooled nothing. With
     * `embed_mark_length`, its first component is the input's token count, so inputs can be told
     * apart. */
    std::vector<float> embedding;
    bool embed_mark_length = false;
    /* Manual mode: see the file comment. A cancel finishes the request at once, as an engine
     * frees a cancelled request's blocks before the next step. */
    bool manual = false;
    /* Refuse every submission the way SchedulerBridge refuses one: id 0 and a failed sink, with
     * this status and reason (RAD_E_FULL is the scheduler's "the prompt is longer than the KV
     * cache can ever hold"). */
    bool refuse = false;
    int refuse_status = RAD_E_INVAL;
    const char* refuse_why = "the scheduler refused the request";
    server::SchedMetrics m;
    /* The draft calibration's phase boundary: what this fake answers, and how often it was asked.
     * Default is a serving deployment's answer (no calibration, so 501). */
    server::DraftCalibAck calib_ack{};
    int calib_calls = 0;

    std::mutex mu;
    std::vector<uint64_t> submitted;
    std::vector<Request> seen;           /* every submitted request, in order */
    SamplingParams last_sp;              /* what the last submitted request was to sample with */
    std::vector<std::string> last_stop;  /* ...and the stop strings it was handed */
    std::string last_label;              /* ...its session label */
    std::vector<std::vector<int32_t>> prompts;   /* every prompt submitted, in order */
    int32_t last_max_tokens = 0;         /* ...and its max_tokens */
    std::vector<uint64_t> cancelled;
    std::vector<uint64_t> reaped;
    std::vector<std::string> reap_errors;
    std::map<uint64_t, Sink*> sinks;     /* every issued id's sink */
    std::vector<std::thread> workers;
    std::atomic<uint64_t> next_id{1};

    ~FakeScheduler() override { join_all(); }

    void join_all() {
        std::vector<std::thread> ws;
        {
            std::lock_guard<std::mutex> lk(mu);
            ws.swap(workers);
        }
        for (auto& t : ws) if (t.joinable()) t.join();
    }

    server::DraftCalibAck draft_calib_boundary(int /*timeout_ms*/) override {
        ++calib_calls;
        return calib_ack;
    }

    uint64_t submit(Request&& r) override {
        if (refuse) {
            sink_fail(r.sink, refuse_status, refuse_why);
            return 0;
        }
        uint64_t id = next_id.fetch_add(1);
        void* sink = r.sink;
        const bool embed = r.embed;
        int32_t cap = r.max_tokens;
        const int np = r.sp.n_probs;
        /* Under the lock: over a real socket, submits arrive from several HTTP workers at once. */
        std::lock_guard<std::mutex> lk(mu);
        submitted.push_back(id);
        sinks[id] = static_cast<Sink*>(sink);
        last_sp = r.sp;
        last_stop = r.stop;
        last_label = r.session_label;
        prompts.push_back(r.prompt);
        last_max_tokens = r.max_tokens;
        seen.push_back(r);
        if (manual) return id;
        std::string text = reply_for ? reply_for(r.prompt) : reply;
        bool h = hang;
        const int fst = fail_status;
        const char* fwhy = fail_why;
        const int32_t nd = drafts;
        std::vector<float> vec = embedding;
        if (embed_mark_length && !vec.empty()) vec[0] = (float)r.prompt.size();
        workers.emplace_back([sink, cap, text, h, fst, fwhy, nd, np, embed, vec] {
            sink_admitted(sink);
            sink_prompt_stats(sink, 8, 4);
            if (embed) {
                if (fst) { sink_fail(sink, fst, fwhy); return; }
                if (!vec.empty()) sink_embedding(sink, vec.data(), (int64_t)vec.size());
                sink_push(sink, 'x', nullptr);
                sink_finish(sink, Finish::Length);
                return;
            }
            int32_t n = 0;
            for (char c : text) {
                if (n >= cap) break;
                if (sink_cancelled(sink)) { sink_finish(sink, Finish::Cancelled); return; }
                if (n > 0 && nd > 0) sink_drafts(sink, nd, 1);
                /* Logprobs when asked: the n-th token at -(n+1)/4, and alternatives 'x', 'y', ...
                 * at -1, -2, ... */
                TokenLogprobs lp;
                lp.token = (int32_t)(unsigned char)c;
                lp.logprob = -0.25f * (float)(n + 1);
                lp.n_top = np > 0 ? np : 0;
                for (int j = 0; j < lp.n_top; ++j) {
                    lp.top_id[j] = 'x' + j;
                    lp.top_lp[j] = -1.0f - (float)j;
                }
                if (!sink_push(sink, (int32_t)(unsigned char)c, np >= 0 ? &lp : nullptr)) return;
                ++n;
                if (h && n >= 3) break;
            }
            if (h) {
                for (int i = 0; i < 20000; ++i) {
                    if (sink_cancelled(sink)) { sink_finish(sink, Finish::Cancelled); return; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                sink_finish(sink, Finish::Length);
                return;
            }
            if (fst) { sink_fail(sink, fst, fwhy); return; }
            sink_finish(sink, n >= cap ? Finish::Length : Finish::Stop);
        });
        return id;
    }

    void cancel(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu);
        cancelled.push_back(id);
        if (!manual) return;
        auto it = sinks.find(id);
        if (it != sinks.end() && !it->second->done()) it->second->finish(Finish::Cancelled);
    }

    /* The contract IScheduler::reap states, checked as it is exercised. */
    void reap(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu);
        if (std::find(reaped.begin(), reaped.end(), id) != reaped.end())
            reap_errors.push_back("id " + std::to_string(id) + " reaped twice");
        auto it = sinks.find(id);
        if (it == sinks.end())
            reap_errors.push_back("id " + std::to_string(id) + " was never issued");
        else if (!it->second->done())
            reap_errors.push_back("id " + std::to_string(id) + " reaped before its sink finished");
        reaped.push_back(id);
    }

    server::SchedMetrics metrics() const override { return m; }

    bool saw_cancel() {
        std::lock_guard<std::mutex> lk(mu);
        return !cancelled.empty();
    }

    /* ---- manual mode: the test is the producer */
    Sink* sink(size_t i) {
        std::lock_guard<std::mutex> lk(mu);
        return i < seen.size() ? static_cast<Sink*>(seen[i].sink) : nullptr;
    }
    void push(size_t i, const std::string& bytes) {
        Sink* s = sink(i);
        for (unsigned char c : bytes) s->push((int32_t)c);
    }
    void finish(size_t i, Finish f) { sink(i)->finish(f); }

    /* After the server is gone: every id it was given was reaped exactly once, after its sink
     * finished. */
    void check_reaped() {
        std::lock_guard<std::mutex> lk(mu);
        for (const std::string& e : reap_errors) fprintf(stderr, "    reap: %s\n", e.c_str());
        CHECK(reap_errors.empty());
        const std::set<uint64_t> want(submitted.begin(), submitted.end());
        const std::multiset<uint64_t> got(reaped.begin(), reaped.end());
        CHECK_EQ(got.size(), want.size());
        CHECK(std::set<uint64_t>(reaped.begin(), reaped.end()) == want);
    }
};

/* Polls `cond` every millisecond for at most `ms`; true once it holds. For states another thread
 * reaches, where no event marks the moment. */
template <class F>
inline bool eventually(F cond, int ms = 3000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!cond()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

/* ================================================================== a server wired to the fakes */

struct Fixture {
    FakeScheduler    sched;
    FakeTokenizer    tok;
    FakeChatTemplate tmpl;
    FakeGrammar      gram;
    std::unique_ptr<Server> srv;

    explicit Fixture(bool with_tools = true, int64_t max_ctx = 0, const char* model_path = "",
                     bool logprobs = false) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        d.grammar = &gram;
        /* The parser travels with the render, so "no tool parsing" is a template returning no
         * parser rather than a null server-wide dependency. */
        tmpl.with_parser = with_tools;
        ServerOptions o;
        o.model_id = "test-model";
        o.default_max_tokens = 32;
        o.max_seqs = 4;
        o.max_ctx = max_ctx;
        o.model_path = model_path;
        o.supports_logprobs = logprobs;
        srv.reset(new Server(d, o));
    }

    /* The same wiring, then `tweak` adjusts the options and dependencies before the server is
     * built. A struct rather than a bare function, which a captureless lambda would convert to
     * the bool of the constructor above. */
    struct Tweak {
        std::function<void(ServerOptions&, Deps&)> fn;
    };
    explicit Fixture(const Tweak& tweak) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        d.grammar = &gram;
        tmpl.with_parser = true;
        ServerOptions o;
        o.model_id = "test-model";
        o.default_max_tokens = 32;
        o.max_seqs = 4;
        tweak.fn(o, d);
        srv.reset(new Server(d, o));
    }

    /* Members are destroyed in reverse declaration order, which would tear down the Server while
     * the fake's worker threads still write into sinks it owns. The engine has the same
     * obligation: stop the scheduler, then destroy the server (server.h). Whatever the case did,
     * every request it submitted was reaped once. */
    ~Fixture() {
        sched.join_all();
        srv.reset();
        sched.check_reaped();
    }

    OaiDeps odeps() {
        OaiDeps d;
        d.tok = &tok;
        d.chat = &tmpl;
        d.grammar = &gram;
        return d;
    }
    OaiLimits limits() {
        OaiLimits l;
        l.model_id = "test-model";
        l.default_max_tokens = 32;
        return l;
    }

    HttpResponse post(const char* path, const std::string& body) {
        return srv->router().call("POST", path, body);
    }
    HttpResponse get(const char* path) { return srv->router().call("GET", path); }
};

/* Drain a streaming response the way the transport does. */
inline std::string drain(HttpResponse& res, int max_chunks = 100000) {
    std::string all;
    for (int i = 0; i < max_chunks; ++i) {
        std::string c;
        bool more = res.next(c);
        all += c;
        if (!more) break;
    }
    if (res.on_close) res.on_close(true);
    return all;
}

/* Split an SSE body into its `data:` payloads and check the framing while doing it. */
inline std::vector<std::string> sse_payloads(const std::string& body, int* failures_at_line) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < body.size()) {
        size_t end = body.find("\n\n", i);
        if (end == std::string::npos) { *failures_at_line = __LINE__; break; }
        std::string ev = body.substr(i, end - i);
        if (ev.rfind("data: ", 0) != 0) { *failures_at_line = __LINE__; break; }
        out.push_back(ev.substr(6));
        i = end + 2;
    }
    return out;
}

/* `j[k]` when it is a string, else "": an error's `param` is null when the error is about no one
 * field, and json::value() throws on a null. */
inline std::string str_at(const json& j, const char* k) {
    if (!j.is_object()) return std::string();
    auto it = j.find(k);
    return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}

/* The JSON of every `data:` line of an event stream with named events (Anthropic's, the Responses
 * API's, the images API's), in order; anything not JSON is skipped. */
inline std::vector<json> sse_events(const std::string& body) {
    std::vector<json> out;
    for (size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const size_t nl = body.find('\n', at);
        json ev = json::parse(body.substr(at + 6, nl - at - 6), nullptr, false);
        at = nl;
        if (ev.is_object()) out.push_back(std::move(ev));
    }
    return out;
}
