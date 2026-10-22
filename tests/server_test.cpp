/* tests/server_test.cpp -- the HTTP layer, driven in process.
 *
 * Router::dispatch is the call the transport makes, so every handler runs exactly as for a real
 * request; only the engine behind it is faked (server_fakes.h), which keeps cancellation and the
 * streaming wire format deterministic. No port is bound; transport_test.cpp covers the socket.
 */
#include "server_fakes.h"

#include "lora/slots.h"
#include "server/liveview.h"
#include "server/metrics.h"
#include "text/chat.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <random>

/* ================================================================== parsing */

TEST(oai_parses_a_full_chat_request_into_sampling_params) {
    Fixture f;
    const std::string body = R"({
      "model":"test-model",
      "messages":[{"role":"system","content":"be brief"},
                  {"role":"user","content":[{"type":"text","text":"hi"}]}],
      "temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,"typical_p":0.95,
      "presence_penalty":0.5,"frequency_penalty":-0.5,
      "repetition_penalty":1.1,"repeat_last_n":128,
      "seed":12345,"stop":["\n\n","END"],"max_tokens":64,"n":2,"stream":false,
      "ignore_eos":true,
      "xtc_probability":0.2,"xtc_threshold":0.15,
      "dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":3,
      "dry_penalty_last_n":256,"dry_sequence_breakers":["\n"],
      "response_format":{"type":"json_schema",
                         "json_schema":{"name":"x","schema":{"type":"object"}}},
      "tools":[{"type":"function","function":{"name":"get_weather",
                "parameters":{"type":"object"}}}],
      "tool_choice":"auto",
      "user":"u-1"
    })";

    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(body, f.odeps(), f.limits(), &r, &e));

    CHECK_NEAR(r.sp.temp, 0.7f, 1e-6);
    CHECK_NEAR(r.sp.top_p, 0.9f, 1e-6);
    CHECK_EQ(r.sp.top_k, 40);
    CHECK_NEAR(r.sp.min_p, 0.05f, 1e-6);
    CHECK_NEAR(r.sp.typical_p, 0.95f, 1e-6);
    CHECK_NEAR(r.sp.pres_penalty, 0.5f, 1e-6);
    CHECK_NEAR(r.sp.freq_penalty, -0.5f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.1f, 1e-6);
    CHECK_EQ(r.sp.penalty_last_n, 128);
    CHECK_NEAR(r.sp.xtc_probability, 0.2f, 1e-6);
    CHECK_NEAR(r.sp.xtc_threshold, 0.15f, 1e-6);
    CHECK_NEAR(r.sp.dry_multiplier, 0.8f, 1e-6);
    CHECK_EQ(r.sp.dry_allowed_length, 3);
    CHECK_EQ(r.sp.dry_penalty_last_n, 256);
    CHECK_EQ((int)r.sp.dry_seq_breakers.size(), 1);
    CHECK_EQ((long long)r.sp.seed, 12345LL);
    CHECK_EQ(r.max_tokens, 64);
    CHECK_EQ(r.n, 2);
    CHECK_EQ((int)r.stop.size(), 2);
    CHECK(r.ignore_eos);
    CHECK(!r.sp.grammar.empty());          /* response_format compiled to a grammar */
    CHECK(r.tools_present);
    CHECK(r.parse_tools);
    CHECK_EQ((int)r.prompts.size(), 1);
    CHECK_EQ(r.n_choices(), 2);
    CHECK(r.id.rfind("chatcmpl-", 0) == 0);
    /* The template ran and the prompt tokenised. */
    CHECK(r.prompts[0].text.find("be brief") != std::string::npos);
    CHECK(!r.prompts[0].tokens.empty());
}

TEST(oai_parses_a_minimal_chat_request_with_defaults) {
    Fixture f;
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})",
                                f.odeps(), f.limits(), &r, &e));
    CHECK_NEAR(r.sp.temp, 1.0f, 1e-6);
    CHECK_NEAR(r.sp.top_p, 1.0f, 1e-6);
    CHECK_EQ(r.sp.top_k, 0);
    CHECK_NEAR(r.sp.min_p, 0.0f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.0f, 1e-6);
    CHECK_EQ(r.n, 1);
    CHECK_EQ(r.max_tokens, 32);           /* the server default, not zero */
    CHECK_EQ((int)r.stop.size(), 0);
    CHECK(r.sp.grammar.empty());
    CHECK(!r.stream);
    CHECK(!r.tools_present);
    CHECK_EQ(r.model, std::string("test-model"));
    /* llama-server's DRY breakers when none are named; a list that is sent replaces them. */
    CHECK(r.sp.dry_seq_breakers == std::vector<std::string>({ "\n", ":", "\"", "*" }));
    OaiRequest none;
    CHECK_OK(parse_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":[]})",
        f.odeps(), f.limits(), &none, &e));
    CHECK(none.sp.dry_seq_breakers.empty());
}

/* The deployment's sampler reaches a request that sends none, and loses to one that does, field
 * by field. The built-in defaults are untruncated, which makes a checkpoint tuned on a truncated
 * tail end replies early (end-of-turn lives in the tail); OaiLimits::default_sampling states the
 * sampler the checkpoint asks for. */
TEST(oai_seeds_sampling_from_the_deployment_default) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_sampling.temp  = 0.7f;
    lim.default_sampling.top_k = 20;
    lim.default_sampling.top_p = 0.95f;
    lim.default_sampling.min_p = 0.01f;

    /* Nothing sent: every field is the deployment's. */
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})",
                                f.odeps(), lim, &r, &e));
    CHECK_NEAR(r.sp.temp, 0.7f, 1e-6);
    CHECK_EQ(r.sp.top_k, 20);
    CHECK_NEAR(r.sp.top_p, 0.95f, 1e-6);
    CHECK_NEAR(r.sp.min_p, 0.01f, 1e-6);

    /* One field sent: that one is the caller's and the rest are still the deployment's; a
     * default that could not be overridden would leave the caller unable to steer. */
    OaiRequest r2;
    CHECK_OK(parse_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"temperature":0,"top_k":0})",
        f.odeps(), lim, &r2, &e));
    CHECK_NEAR(r2.sp.temp, 0.0f, 1e-6);
    CHECK_EQ(r2.sp.top_k, 0);              /* 0 is a value, not silence */
    CHECK_NEAR(r2.sp.top_p, 0.95f, 1e-6);
    CHECK(r2.sp.greedy());

    /* The same on /v1/completions: one model served two ways must sample the same. */
    OaiRequest r3;
    CHECK_OK(parse_completion_request(R"({"prompt":"ab"})", f.odeps(), lim, &r3, &e));
    CHECK_NEAR(r3.sp.temp, 0.7f, 1e-6);
    CHECK_EQ(r3.sp.top_k, 20);
    CHECK_NEAR(r3.sp.top_p, 0.95f, 1e-6);
}

TEST(oai_parses_a_completion_prompt_in_every_shape) {
    Fixture f;
    OaiLimits lim = f.limits();
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"ab"})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 1);
        CHECK_EQ((int)r.prompts[0].tokens.size(), 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":["a","bc"]})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 2);
        CHECK_EQ(r.n_choices(), 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[104,105]})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 1);
        CHECK_EQ(r.prompts[0].text, std::string("hi"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[[104],[105,106]]})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 2);
    }
}

/* A max_tokens that does not fit is clamped, not refused. The field is an upper bound and the
 * context a lower one; refusing produces no tokens where clamping produces every token that could
 * have been produced. A prompt with no room left to generate into stays a refusal, since it is
 * not servable at any max_tokens. */
TEST(a_max_tokens_past_the_context_is_clamped_to_what_is_left) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_ctx = 64;

    /* Chat. The assertion is against the rendered length rather than a constant: the rendered
     * prompt is not the content that went in, and a hard-coded difference would break on the
     * next change to the fake template. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_chat_request(
            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":9999})",
            f.odeps(), lim, &r, &e));
        CHECK_EQ((int64_t)r.prompts[0].tokens.size() + r.max_tokens, lim.max_ctx);
    }

    /* A request that already fits is left exactly as it asked. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"ab","max_tokens":3})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.max_tokens, 3);
    }

    /* A batch clamps against its longest prompt, not its last: one max_tokens covers them all. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":["a","abcde"],"max_tokens":9999})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.max_tokens, 64 - 5);
    }

    /* A prompt that does not fit on its own is still a 400 that says so. */
    {
        OaiRequest r; ApiError e;
        std::string body = R"({"prompt":")" + std::string(70, 'x') + R"("})";
        CHECK_EQ(parse_completion_request(body, f.odeps(), lim, &r, &e), RAD_E_INVAL);
        CHECK_EQ(e.code, std::string("context_length_exceeded"));
    }
}

/* A max_tokens past 32 bits is refused, not narrowed. Narrowed, 2^31 is negative and 2^32 is zero,
 * and the scheduler reads either as "no limit", so a requested bound would vanish. These limits
 * declare no context, so nothing downstream could clamp it back. */
TEST(a_max_tokens_past_32_bits_is_refused_rather_than_wrapped) {
    Fixture f;
    OaiLimits lim = f.limits();

    for (const char* v : {"2147483648", "4294967296", "4294967297", "18446744073709551615"}) {
        OaiRequest r; ApiError e;
        std::string body = std::string(R"({"prompt":"a","max_tokens":)") + v + "}";
        CHECK(parse_completion_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("max_tokens"));
    }
    /* Both spellings: the chat endpoint's newer one names itself. */
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],
                                     "max_completion_tokens":2147483648})",
                                 f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("max_completion_tokens"));
    }
    /* The largest value that fits is still a value. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"a","max_tokens":2147483647})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ(r.max_tokens, (int32_t)2147483647);
    }
    /* Over HTTP it is the 400, not an admitted request. */
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"a","max_tokens":4294967296})").status, 400);
    CHECK(f.sched.submitted.empty());
}

/* A token id must fit an int32 or it is refused. Narrowed, 2^32 + 5 would silently be token 5. */
TEST(a_token_id_outside_int32_is_refused_rather_than_aliased) {
    Fixture f;
    OaiLimits lim = f.limits();
    for (const char* body : {R"({"prompt":[104,4294967301]})", R"({"prompt":[-1]})",
                             R"({"prompt":[[104],[2147483648]]})",
                             R"({"prompt":[18446744073709551615]})"}) {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("prompt"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":[4294967301]})", f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("input"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[104,2147483647]})", f.odeps(), lim, &r,
                                          &e));
        REQUIRE_EQ(r.prompts.size(), (size_t)1);
        REQUIRE_EQ(r.prompts[0].tokens.size(), (size_t)2);
        CHECK_EQ(r.prompts[0].tokens[1], (int32_t)2147483647);
    }
}

/* A field of the wrong type is the caller's mistake, so it is a 400 naming it, never an exception
 * out of the JSON library, which the transport can only report as a 500. */
TEST(a_wrongly_typed_field_is_a_400_not_an_exception) {
    Fixture f;
    FakeMultimodal mm;
    OaiDeps d = f.odeps();
    d.mm = &mm;
    OaiLimits lim = f.limits();
    lim.allow_image = true;
    lim.allow_audio = true;

    const char* bodies[] = {
        R"({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":5,"function":{"name":"f"}}]})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":7}}]})",
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":5}}]}]})",
        R"({"messages":[{"role":"user","content":[{"type":"input_audio",
            "input_audio":{"data":"AAAA","format":5}}]}]})",
        R"({"messages":[{"role":"user","content":[{"type":"input_audio",
            "input_audio":{"data":5,"format":"wav"}}]}]})",
    };
    for (const char* body : bodies) {
        OaiRequest r; ApiError e;
        int rc = RAD_OK;
        bool threw = false;
        try { rc = parse_chat_request(body, d, lim, &r, &e); } catch (...) { threw = true; }
        CHECK(!threw);
        CHECK(rc < 0);
        CHECK_EQ(e.status, 400);
    }
}

/* Every stop string is scanned for in every stretch of output, so the list is bounded in count
 * and length, and a list past either bound is refused rather than paid for on every token. */
TEST(a_stop_list_past_its_bounds_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    auto body_with = [](size_t n, size_t len) {
        json b;
        b["prompt"] = "a";
        b["stop"] = json::array();
        for (size_t i = 0; i < n; ++i) b["stop"].push_back(std::string(len, (char)('a' + i % 26)));
        return b.dump();
    };
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(body_with(64, 4096), f.odeps(), lim, &r, &e));
        CHECK_EQ(r.stop.size(), (size_t)64);
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body_with(65, 1), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body_with(1, 4097), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
    {
        OaiRequest r; ApiError e;
        const std::string one = R"({"prompt":"a","stop":")" + std::string(4097, 'z') + R"("})";
        CHECK(parse_completion_request(one, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
}

/* The parsed tree is bounded, not only the body: `[1,1,...]` is two bytes per value on the wire
 * and several times that once parsed. One value past the budget is refused, on the generation
 * endpoints and on /detokenize, whose body is exactly such an array. */
TEST(a_body_holding_more_values_than_the_budget_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    const int64_t budget = json_value_budget(lim.max_ctx);
    CHECK(json_value_budget(1000) > budget);   /* it scales with the context */

    /* The object and the array are values too, so `inner` scalars make inner + 2 in all. */
    auto ids = [](const char* key, int64_t inner) {
        std::string s = std::string("{\"") + key + "\":[";
        s.reserve((size_t)inner * 2 + 32);
        for (int64_t i = 0; i < inner; ++i) { if (i) s += ','; s += '7'; }
        return s + "]}";
    };
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(ids("prompt", budget - 2), f.odeps(), lim, &r, &e));
        REQUIRE_EQ(r.prompts.size(), (size_t)1);
        CHECK_EQ((int64_t)r.prompts[0].tokens.size(), budget - 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(ids("prompt", budget - 1), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.message, "JSON values"));
    }
    CHECK_EQ(f.post("/detokenize", ids("tokens", budget - 1)).status, 400);
    CHECK_EQ(f.post("/detokenize", ids("tokens", 3)).status, 200);
}

/* A serialiser recurses once per nesting level, so depth is bounded separately from the value
 * count: a few hundred kilobytes of brackets is far inside the value budget and far past a
 * worker's stack. The bound covers the body and the strings the chat template parses as JSON. */
TEST(a_body_or_template_json_nested_past_the_bound_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    auto nest = [](int depth) {
        return std::string((size_t)depth, '[') + "1" + std::string((size_t)depth, ']');
    };
    auto quoted = [](const std::string& s) { return "\"" + s + "\""; };
    const int deep = 200000, fine = kJsonMaxDepth - 8;
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request("{\"prompt\":\"x\",\"logit_bias\":" + nest(deep) + "}",
                                       f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.message, "levels deep"));
    }
    CHECK_EQ(f.post("/detokenize", "{\"tokens\":" + nest(deep) + "}").status, 400);

    auto call = [&](const std::string& args) {
        return R"({"messages":[{"role":"user","content":"hi"},
                   {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"f","arguments":)" + quoted(args) + R"(}}]},
                   {"role":"tool","tool_call_id":"c1","content":"ok"}]})";
    };
    auto result = [&](const std::string& content) {
        return R"({"messages":[{"role":"user","content":"hi"},
                   {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"f","arguments":"{}"}}]},
                   {"role":"tool","tool_call_id":"c1","content":)" + quoted(content) + "}]}";
    };
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(call(nest(deep)), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.param, "arguments"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(result(nest(deep)), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
    }
    /* Brackets inside a string literal of the argument text are not nesting. */
    {
        OaiRequest r; ApiError e;
        const std::string in_str = "{\\\"s\\\":\\\"" + std::string(5000, '[') + "\\\"}";
        CHECK_OK(parse_chat_request(call(in_str), f.odeps(), lim, &r, &e));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_chat_request(call(nest(fine)), f.odeps(), lim, &r, &e));
        CHECK_OK(parse_chat_request(result(nest(fine)), f.odeps(), lim, &r, &e));
    }
    /* A tool result that is not JSON is never parsed into a tree, so brackets in it are text:
     * openers that never close, and nesting with text around it. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_chat_request(result(std::string(5000, '[')), f.odeps(), lim, &r, &e));
        CHECK_OK(parse_chat_request(result("got: " + nest(deep)), f.odeps(), lim, &r, &e));
        CHECK_OK(parse_chat_request(result(nest(deep) + ","), f.odeps(), lim, &r, &e));
    }
}

/* ================================================================== the 400 path */

TEST(unsupported_parameters_are_400_and_name_themselves) {
    Fixture f;

    /* A parameter we know about and do not implement. */
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "logit_bias":{"5":10}})");
    CHECK_EQ(r.status, 400);
    json e = json::parse(r.body);
    CHECK_EQ(e["error"]["param"].get<std::string>(), std::string("logit_bias"));
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("unsupported_parameter"));
    CHECK(e["error"]["message"].get<std::string>().find("logit_bias") != std::string::npos);

    /* An unknown parameter is a client's own extension and is ignored, as llama.cpp and vLLM
     * ignore it; refusing one (e.g. OpenRouter's session_id) breaks clients. */
    HttpResponse r2 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "wobble":1,"session_id":"abc"})");
    CHECK_EQ(r2.status, 200);

    /* What llama.cpp-built clients send on every request: samplers this server does not run, at
     * their off values. Accepted, and refused by name once one is switched on. */
    HttpResponse off = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "mirostat":0,"top_n_sigma":-1,"dynatemp_range":0,
                                 "logit_bias":{},"cache_prompt":true,"id_slot":-1,
                                 "backend_sampling":false,"speculative.n_max":16})");
    CHECK_EQ(off.status, 200);
    HttpResponse on = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],"mirostat":2})");
    CHECK_EQ(on.status, 400);
    CHECK_EQ(json::parse(on.body)["error"]["param"].get<std::string>(), std::string("mirostat"));

    /* Out-of-range values name the field and the range. */
    HttpResponse r3 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "temperature":9})");
    CHECK_EQ(r3.status, 400);
    CHECK(r3.body.find("temperature") != std::string::npos);

    /* logprobs without sampler support is refused rather than answered with nulls. */
    HttpResponse r5 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "logprobs":true})");
    CHECK_EQ(r5.status, 400);
    CHECK(r5.body.find("logprobs") != std::string::npos);

    /* A body that is not an object at all. */
    HttpResponse r6 = f.post("/v1/chat/completions", "[]");
    CHECK_EQ(r6.status, 400);
}

TEST(unknown_routes_and_methods_say_which) {
    Fixture f;
    HttpResponse r = f.get("/v1/nope");
    CHECK_EQ(r.status, 404);
    CHECK(r.body.find("/v1/nope") != std::string::npos);
    HttpResponse r2 = f.get("/v1/chat/completions");
    CHECK_EQ(r2.status, 405);
}

/* ================================================================== non-streaming shape */

TEST(non_streaming_chat_response_has_the_openai_shape) {
    Fixture f;
    f.sched.reply = "hello";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(!r.streaming());

    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("chat.completion"));
    CHECK(j["id"].get<std::string>().rfind("chatcmpl-", 0) == 0);
    CHECK_EQ(j["model"].get<std::string>(), std::string("test-model"));
    CHECK(j["created"].is_number_integer());
    CHECK_EQ((int)j["choices"].size(), 1);

    const auto& c = j["choices"][0];
    CHECK_EQ(c["index"].get<int>(), 0);
    CHECK_EQ(c["message"]["role"].get<std::string>(), std::string("assistant"));
    CHECK_EQ(c["message"]["content"].get<std::string>(), std::string("hello"));
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(c["logprobs"].is_null());

    CHECK_EQ(j["usage"]["completion_tokens"].get<int>(), 5);
    CHECK(j["usage"]["prompt_tokens"].get<int>() > 0);
    CHECK_EQ(j["usage"]["total_tokens"].get<int>(),
             j["usage"]["prompt_tokens"].get<int>() + 5);
    CHECK(j["usage"]["prompt_tokens_details"]["cached_tokens"].is_number());
}

TEST(non_streaming_completion_response_has_the_openai_shape) {
    Fixture f;
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","max_tokens":8})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("text_completion"));
    CHECK_EQ(j["choices"][0]["text"].get<std::string>(), std::string("abc"));
    CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(j["choices"][0]["logprobs"].is_null());
}

/* A generation finishes on its own thread while the request's thread drains it, so the finish can
 * land between a drain and the check after it; a choice closed there loses the tokens pushed just
 * before the finish (the reply cut short, a stop string in them unseen). The window holds the
 * stop-string scan, so the requests carry stop strings the reply never contains, four run at a
 * time, and every reply must arrive whole. Intermittent by nature: with the finish read after the
 * drain this cut replies short within a few hundred requests. */
TEST(a_reply_finishing_during_a_drain_keeps_its_last_tokens) {
    Fixture f;
    std::string reply;
    for (int k = 0; k < 24; ++k) reply += "the reply goes on, " + std::to_string(k) + "; ";
    f.sched.reply = reply;
    std::atomic<int> cut{0}, bad{0};
    std::vector<std::thread> posters;
    for (int t = 0; t < 4; ++t)
        posters.emplace_back([&] {
            for (int i = 0; i < 150; ++i) {
                HttpResponse r = f.post("/v1/completions",
                    R"({"prompt":"hi","max_tokens":1000,"stop":["@@@@","####","%%%%"]})");
                if (r.status != 200) { ++bad; continue; }
                if (json::parse(r.body)["choices"][0]["text"].get<std::string>() != reply) ++cut;
            }
        });
    for (auto& t : posters) t.join();
    f.sched.join_all();
    if (cut) std::fprintf(stderr, "    %d of 600 replies cut short\n", cut.load());
    CHECK_EQ(bad.load(), 0);
    CHECK_EQ(cut.load(), 0);
}

TEST(n_greater_than_one_fans_out_into_one_choice_each) {
    Fixture f;
    f.sched.reply = "xy";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"n":3})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ((int)j["choices"].size(), 3);
    CHECK_EQ(j["choices"][2]["index"].get<int>(), 2);
    CHECK_EQ((int)f.sched.submitted.size(), 3);
}

TEST(a_stop_string_is_cut_out_of_the_answer) {
    Fixture f;
    f.sched.reply = "abENDcd";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(), std::string("ab"));
    CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
    /* Cutting the answer frees the blocks immediately rather than at completion (spec §14). */
    CHECK(f.sched.saw_cancel());
}

TEST(a_complete_tool_call_is_reported_as_tool_calls) {
    Fixture f;
    f.sched.reply = "TOOL:get_weather:{\"city\":\"berlin\"}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    const auto& c = j["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("tool_calls"));
    CHECK_EQ((int)c["message"]["tool_calls"].size(), 1);
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["name"].get<std::string>(),
             std::string("get_weather"));
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["arguments"].get<std::string>(),
             std::string("{\"city\":\"berlin\"}"));
    CHECK(c["message"]["content"].is_null());
}

/* The three spellings of "do not think" must all reach the template, and the kwarg must reach
 * both places: the lift overwrites the context key `enable_thinking` with its own field
 * (chat-auto-parser-helpers.cpp), so a kwarg not also lifted into the field would be discarded
 * and the caller would get thinking despite asking for the opposite. */
TEST(the_ways_of_turning_thinking_off_all_reach_the_template) {
    for (const char* body : {
             R"({"messages":[{"role":"user","content":"hi"}],
                 "chat_template_kwargs":{"enable_thinking":false}})",
             R"({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false})",
             R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"})" }) {
        Fixture f;
        CHECK_EQ(f.post("/v1/chat/completions", body).status, 200);
        CHECK(!f.tmpl.last_opt.enable_thinking);
    }
    /* ...and the default is left alone. */
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);
    CHECK(f.tmpl.last_opt.enable_thinking);
}

/* ================================================================== deployment defaults
 *
 * Every default the command line can move, seen from a request: it reaches a request that leaves
 * the field out, and loses to one that names it. */

static OaiRequest chat_with(Fixture& f, const OaiLimits& lim, const std::string& extra) {
    OaiRequest r;
    ApiError e;
    const std::string body =
        R"({"messages":[{"role":"user","content":"hi"}])" + (extra.empty() ? "" : "," + extra) + "}";
    CHECK_OK(parse_chat_request(body, f.odeps(), lim, &r, &e));
    return r;
}

TEST(the_rest_of_the_sampler_is_seeded_from_the_deployment_too) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_sampling.pres_penalty = 1.5f;
    lim.default_sampling.rep_penalty = 1.05f;
    lim.default_sampling.penalty_last_n = 256;
    lim.default_sampling.typical_p = 0.9f;
    lim.default_sampling.dry_multiplier = 0.8f;
    lim.default_sampling.xtc_probability = 0.5f;

    OaiRequest r = chat_with(f, lim, "");
    CHECK_NEAR(r.sp.pres_penalty, 1.5f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.05f, 1e-6);
    CHECK_EQ(r.sp.penalty_last_n, 256);
    CHECK_NEAR(r.sp.typical_p, 0.9f, 1e-6);
    CHECK_NEAR(r.sp.dry_multiplier, 0.8f, 1e-6);
    CHECK_NEAR(r.sp.xtc_probability, 0.5f, 1e-6);

    OaiRequest r2 = chat_with(f, lim, R"("presence_penalty":0,"repeat_last_n":64)");
    CHECK_NEAR(r2.sp.pres_penalty, 0.0f, 1e-6);
    CHECK_EQ(r2.sp.penalty_last_n, 64);
    CHECK_NEAR(r2.sp.rep_penalty, 1.05f, 1e-6);

    /* A stated empty breaker list stays empty, which SamplingParams alone cannot express (unstated
     * means llama-server's four). A request's own list still wins. */
    lim.default_sampling.dry_seq_breakers.clear();
    lim.default_dry_breakers_set = true;
    CHECK(chat_with(f, lim, "").sp.dry_seq_breakers.empty());
    CHECK(chat_with(f, lim, R"("dry_sequence_breakers":["x"])").sp.dry_seq_breakers ==
          std::vector<std::string>({ "x" }));
    lim.default_sampling.dry_seq_breakers = { "\n" };
    CHECK(chat_with(f, lim, "").sp.dry_seq_breakers == std::vector<std::string>({ "\n" }));
}

/* A sampler field's range is one table for the request and the flag; this pins the request half
 * to the exact sentences requests get. */
TEST(the_shared_sampler_ranges_refuse_in_the_words_a_request_always_got) {
    struct Case { const char* body; const char* param; const char* msg; };
    const Case cases[] = {
        { R"("temperature":2.5)",        "temperature",        "must be in [0, 2], got 2.5" },
        { R"("top_p":0)",                "top_p",              "must be in (0, 1]" },
        { R"("top_k":-1)",               "top_k",              "must be >= 0 (0 samples from the "
                                                               "1024 most likely tokens, the "
                                                               "sampler's candidate width)" },
        { R"("min_p":1.5)",              "min_p",              "must be in [0, 1], got 1.5" },
        { R"("typical_p":0)",            "typical_p",          "must be in (0, 1]" },
        { R"("presence_penalty":3)",     "presence_penalty",   "must be in [-2, 2], got 3" },
        { R"("frequency_penalty":-3)",   "frequency_penalty",  "must be in [-2, 2], got -3" },
        { R"("repetition_penalty":0)",   "repetition_penalty", "must be > 0" },
        { R"("repeat_last_n":-2)",       "repeat_last_n",      "must be >= -1" },
        { R"("xtc_probability":2)",      "xtc_probability",    "must be in [0, 1], got 2" },
        { R"("xtc_threshold":-1)",       "xtc_threshold",      "must be in [0, 1], got -1" },
        { R"("dry_base":0.5)",           "dry_base",           "must be >= 1" },
        { R"("dry_allowed_length":0)",   "dry_allowed_length", "must be >= 1" },
        { R"("dry_penalty_last_n":-2)",  "dry_penalty_last_n", "must be >= -1" },
    };
    Fixture f;
    for (const Case& c : cases) {
        OaiRequest r;
        ApiError e;
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":"hi"}],)") + c.body + "}";
        CHECK_EQ(parse_chat_request(body, f.odeps(), f.limits(), &r, &e), RAD_E_INVAL);
        CHECK_EQ(e.param, std::string(c.param));
        CHECK_EQ(e.message, std::string(c.param) + ": " + c.msg);
    }
    /* The other half: the same function, as config.cpp calls it. */
    std::string why;
    CHECK(sampler_value_ok("presence_penalty", -2.0, &why));
    CHECK(!sampler_value_ok("presence_penalty", std::nan(""), &why));
    CHECK(!sampler_value_ok("temperature", std::nan(""), &why));
    CHECK(sampler_value_ok("dry_multiplier", -5.0, &why));     /* unbounded, as a request is */
    /* A DRY window under one token, which the sampler would refuse at admission, is refused for
     * the flag as for a request; one is the least. */
    CHECK(!sampler_value_ok("dry_allowed_length", 0.0, &why));
    CHECK(sampler_value_ok("dry_allowed_length", 1.0, &why));
}

/* `--default-max-tokens auto` is what the context leaves, and `--max-tokens-cap` clamps as the
 * context does: a request asking for more is served up to the cap, not refused. */
TEST(the_max_tokens_default_and_cap_bound_a_request_like_the_context) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_ctx = 64;
    lim.default_max_tokens = 0;

    OaiRequest r = chat_with(f, lim, "");
    CHECK_EQ((int64_t)r.prompts[0].tokens.size() + r.max_tokens, lim.max_ctx);
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":5)").max_tokens, 5);

    lim.max_tokens_cap = 10;
    CHECK_EQ(chat_with(f, lim, "").max_tokens, 10);                     /* auto, then the cap */
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":9999)").max_tokens, 10);
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":5)").max_tokens, 5);

    /* `auto` is the default: limits that say nothing about max_tokens give a request naming none
     * what the context leaves. A fixed small default would end a reasoning model's answers inside
     * its reasoning, with a finish_reason "length" chat clients do not show. */
    {
        OaiLimits d;
        d.model_id = lim.model_id;
        d.max_ctx = 64;
        OaiRequest a = chat_with(f, d, "");
        CHECK_EQ((int64_t)a.prompts[0].tokens.size() + a.max_tokens, d.max_ctx);
    }

    /* The cap holds where no context is declared, and on /v1/completions. */
    lim.max_ctx = 0;
    lim.default_max_tokens = 512;
    CHECK_EQ(chat_with(f, lim, "").max_tokens, 10);
    OaiRequest c;
    ApiError e;
    CHECK_OK(parse_completion_request(R"({"prompt":"ab","max_tokens":100})", f.odeps(), lim, &c, &e));
    CHECK_EQ(c.max_tokens, 10);

    /* Unset, nothing moves: the fixture's default with no cap and no context. */
    CHECK_EQ(chat_with(f, f.limits(), "").max_tokens, 32);
    CHECK_EQ(chat_with(f, f.limits(), R"("max_tokens":100000)").max_tokens, 100000);
}

TEST(the_stop_bounds_are_the_deployments) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_stops = 2;
    lim.max_stop_bytes = 3;
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":["a","bcd"]})",
                                f.odeps(), lim, &r, &e));
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":["a","b","c"]})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK(has(e.message, "at most 2 stop strings"));
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":"abcd"})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK(has(e.message, "at most 3 bytes"));
}

/* The deployment's template variables go under the request's, key by key. */
TEST(the_deployments_template_variables_reach_the_template_under_the_requests) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_template_kwargs = { { "custom", "\"a\"" }, { "depth", "2" } };

    chat_with(f, lim, "");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("custom"), std::string("\"a\""));
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("depth"), std::string("2"));

    chat_with(f, lim, R"("chat_template_kwargs":{"custom":"b"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("custom"), std::string("\"b\""));
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("depth"), std::string("2"));

    lim.reasoning_format = "none";
    chat_with(f, lim, "");
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("none"));
    chat_with(f, f.limits(), "");
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("auto"));
    CHECK(f.tmpl.last_opt.template_kwargs.empty());

    /* /tokenize renders with the same variables, so the prompt it counts is the prompt chat
     * serves; with none set it renders unchanged. */
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    const std::string body = R"({"messages":[{"role":"user","content":"hi"}]})";
    {
        ServerOptions o;
        o.default_template_kwargs = { { "enable_thinking", "false" }, { "custom", "\"a\"" } };
        Server srv(d, o);
        CHECK_EQ(srv.router().call("POST", "/tokenize", body).status, 200);
        CHECK(!tmpl.last_opt.enable_thinking);
        CHECK_EQ(tmpl.last_opt.template_kwargs.at("custom"), std::string("\"a\""));
        CHECK_EQ(srv.router().call("POST", "/tokenize",
                                   R"({"messages":[{"role":"user","content":"hi"}],
                                       "chat_template_kwargs":{"enable_thinking":true}})").status, 200);
        CHECK_EQ(tmpl.last_opt.template_kwargs.at("enable_thinking"), std::string("true"));
    }
    {
        ServerOptions o;
        Server srv(d, o);
        CHECK_EQ(srv.router().call("POST", "/tokenize", body).status, 200);
        CHECK(tmpl.last_opt.enable_thinking);
        CHECK(tmpl.last_opt.template_kwargs.empty());
    }
}

/* Thinking is one setting with three spellings, and a deployment default in one spelling must not
 * outlive a request that used another. */
TEST(a_request_that_says_anything_about_thinking_replaces_the_deployments_default) {
    Fixture f;
    OaiLimits off = f.limits();
    off.default_template_kwargs = { { "enable_thinking", "false" } };

    chat_with(f, off, "");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("enable_thinking"), std::string("false"));

    for (const char* extra : { R"("enable_thinking":true)",
                               R"("chat_template_kwargs":{"enable_thinking":true})",
                               R"("reasoning_effort":"high")",
                               R"("chat_template_kwargs":{"reasoning_effort":"high"})" }) {
        chat_with(f, off, extra);
        CHECK(f.tmpl.last_opt.enable_thinking);
    }
    chat_with(f, off, R"("reasoning_effort":"high")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"high\""));
    CHECK(!f.tmpl.last_opt.template_kwargs.count("enable_thinking"));
    /* An empty effort is no effort, and leaves the default standing. */
    chat_with(f, off, R"("reasoning_effort":"")");
    CHECK(!f.tmpl.last_opt.enable_thinking);

    /* --reasoning-effort none is the same default in its other spelling. */
    OaiLimits none = f.limits();
    none.default_reasoning_effort = "none";
    chat_with(f, none, "");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    chat_with(f, none, R"("enable_thinking":true)");
    CHECK(f.tmpl.last_opt.enable_thinking);
    chat_with(f, none, R"("reasoning_effort":"low")");
    CHECK(f.tmpl.last_opt.enable_thinking);

    /* A level is an effort default: a request that only turns thinking on keeps it, and one that
     * names its own effort replaces it. */
    OaiLimits low = f.limits();
    low.default_reasoning_effort = "low";
    chat_with(f, low, R"("enable_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));
    chat_with(f, low, R"("reasoning_effort":"xhigh")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, low, R"("chat_template_kwargs":{"reasoning_effort":"xhigh"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));

    OaiLimits kw_low = f.limits();
    kw_low.default_template_kwargs = { { "reasoning_effort", "\"low\"" } };
    chat_with(f, kw_low, R"("enable_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));
    chat_with(f, kw_low, R"("reasoning_effort":"xhigh")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
}

/* A template default the template refuses is found at startup and attributed to the flag that
 * set it. A render that fails without any default blames nothing. */
TEST(a_template_default_the_template_refuses_is_found_before_a_caller_sends_one) {
    auto run = [](const std::map<std::string, std::string>& kw, const std::string& effort,
                  const std::string& refuse, std::string* kw_why, std::string* eff_why) {
        FakeScheduler sched;
        FakeTokenizer tok;
        FakeChatTemplate tmpl;
        tmpl.refuse_kwarg = refuse;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        ServerOptions o;
        o.default_template_kwargs = kw;
        o.default_reasoning_effort = effort;
        Server srv(d, o);
        srv.check_template_defaults(kw_why, eff_why);
    };
    std::string kw_why, eff_why;
    run({ { "bad", "1" } }, "", "bad", &kw_why, &eff_why);
    CHECK(has(kw_why, "does not take bad"));
    CHECK(eff_why.empty());

    run({ { "fine", "1" } }, "high", "reasoning_effort", &kw_why, &eff_why);
    CHECK(kw_why.empty());
    CHECK(has(eff_why, "does not take reasoning_effort"));

    run({ { "fine", "1" } }, "high", "", &kw_why, &eff_why);
    CHECK(kw_why.empty() && eff_why.empty());
    /* "none" is enable_thinking false, never a template variable, so nothing to refuse. */
    run({}, "none", "reasoning_effort", &kw_why, &eff_why);
    CHECK(kw_why.empty() && eff_why.empty());
}

/* An effort the template refuses is named as one. The template's own set is probed at startup, a
 * request outside it is answered as OpenAI answers an unsupported value (`reasoning_effort`,
 * `unsupported_value`, the set in `supported`), and /server_info publishes the set beforehand,
 * rather than blaming `messages` with the set buried in a Jinja traceback. */
TEST(an_effort_the_template_refuses_is_answered_with_the_set_it_takes) {
    Fixture f;
    f.tmpl.efforts = { "\"xhigh\"", "\"medium\"", "\"low\"" };
    const std::vector<std::string> set = f.srv->probe_reasoning_efforts();
    REQUIRE_EQ(set.size(), 3u);
    CHECK_EQ(set[0], std::string("low"));
    CHECK_EQ(set[1], std::string("medium"));
    CHECK_EQ(set[2], std::string("xhigh"));
    CHECK(f.srv->template_default_effort().empty());   /* no level renders as no level */
    /* A template that writes one level when a request names none, as Qwen3.8's writes xhigh
     * (#113): the probe names it. */
    f.tmpl.default_effort = "\"xhigh\"";
    f.srv->probe_reasoning_efforts();
    CHECK_EQ(f.srv->template_default_effort(), std::string("xhigh"));
    f.tmpl.default_effort.clear();
    f.srv->probe_reasoning_efforts();

    OaiLimits lim = f.limits();
    lim.reasoning_efforts = set;
    for (const char* extra : { R"("reasoning_effort":"high")",
                               R"("chat_template_kwargs":{"reasoning_effort":"high"})" }) {
        OaiRequest r;
        ApiError e;
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":"hi"}],)") + extra + "}";
        CHECK(parse_chat_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("reasoning_effort"));
        CHECK_EQ(e.code, std::string("unsupported_value"));
        CHECK(!has(e.message, "Jinja"));
        CHECK(has(e.message, "'xhigh'"));
        const json env = json::parse(e.dump());
        REQUIRE(env["error"]["supported"].is_array());
        CHECK_EQ(env["error"]["supported"].size(), 5u);
        CHECK_EQ(env["error"]["supported"][0].get<std::string>(), std::string("none"));
        CHECK_EQ(env["error"]["supported"][4].get<std::string>(), std::string("xhigh"));
    }
    chat_with(f, lim, R"("reasoning_effort":"xhigh")");
    chat_with(f, lim, R"("reasoning_effort":"none")");

    const json j = json::parse(f.srv->router().call("GET", "/server_info").body);
    const json& caps = j["capabilities"]["reasoning_efforts"];
    REQUIRE(caps.is_array());
    CHECK_EQ(caps.size(), 5u);
    CHECK_EQ(caps[2].get<std::string>(), std::string("low"));
}

/* A template that takes any word has no set to publish or refuse with, and a template refusing
 * for another reason keeps blaming `messages` with its own sentence. */
TEST(a_template_that_takes_any_effort_publishes_no_set) {
    Fixture f;
    CHECK(f.srv->probe_reasoning_efforts().empty());
    const json j = json::parse(f.srv->router().call("GET", "/server_info").body);
    CHECK(j["capabilities"]["reasoning_efforts"].is_null());

    f.tmpl.refuse_kwarg = "custom";
    OaiLimits lim = f.limits();
    lim.reasoning_efforts = { "low" };
    OaiRequest r;
    ApiError e;
    CHECK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],
                                 "reasoning_effort":"low","chat_template_kwargs":{"custom":1}})",
                             f.odeps(), lim, &r, &e) < 0);
    CHECK_EQ(e.param, std::string("messages"));
    CHECK(has(e.message, "does not take custom"));
}

/* --reasoning-effort-map: the deployment's word replaces the request's before the template sees
 * it, in either spelling; a word mapped to "none" turns thinking off; a target the template
 * refuses is found at startup. */
TEST(a_mapped_effort_reaches_the_template_as_the_deployments_word) {
    Fixture f;
    f.tmpl.efforts = { "\"xhigh\"", "\"medium\"", "\"low\"" };
    OaiLimits lim = f.limits();
    lim.reasoning_effort_map = { { "high", "xhigh" }, { "minimal", "low" }, { "max", "none" } };
    chat_with(f, lim, R"("reasoning_effort":"high")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, lim, R"("chat_template_kwargs":{"reasoning_effort":"high"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, lim, R"("reasoning_effort":"medium")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"medium\""));
    chat_with(f, lim, R"("reasoning_effort":"max")");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    CHECK(!f.tmpl.last_opt.template_kwargs.count("reasoning_effort"));
    /* The map is read before "none" and "minimal" are, so it can name either. */
    chat_with(f, lim, R"("reasoning_effort":"minimal")");
    CHECK(f.tmpl.last_opt.enable_thinking);
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));

    auto check = [&](const std::map<std::string, std::string>& m) {
        FakeScheduler sched;
        FakeTokenizer tok;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &f.tmpl;
        ServerOptions o;
        o.reasoning_effort_map = m;
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK_EQ(j["request_defaults"]["reasoning_effort_map"].size(), m.size());
        return srv.check_reasoning_effort_map();
    };
    CHECK(check({ { "high", "xhigh" }, { "max", "none" } }).empty());
    CHECK(has(check({ { "high", "ultra" } }), "high=ultra"));
}

TEST(server_info_reports_what_a_request_that_leaves_a_field_out_gets) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    {
        ServerOptions o;
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK(j["server"]["default_max_tokens"].is_null());        /* auto */
        CHECK(j["server"]["max_tokens_cap"].is_null());
        CHECK_EQ(j["server"]["max_stop_strings"].get<int64_t>(), 64);
        const json& r = j["request_defaults"];
        CHECK_NEAR(r["temperature"].get<double>(), 1.0, 1e-9);
        CHECK_NEAR(r["repetition_penalty"].get<double>(), 1.0, 1e-9);
        CHECK_EQ(r["dry_sequence_breakers"].size(), 4u);
        CHECK(r["reasoning_effort"].is_null());
        CHECK(r["chat_template_kwargs"].is_object() && r["chat_template_kwargs"].empty());
        CHECK_EQ(r["reasoning_format"].get<std::string>(), std::string("auto"));
    }
    {
        ServerOptions o;
        o.default_max_tokens = 0;
        o.max_tokens_cap = 4096;
        o.default_sampling.pres_penalty = 1.5f;
        o.default_template_kwargs = { { "enable_thinking", "false" }, { "tag", "\"x\"" } };
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK(j["server"]["default_max_tokens"].is_null());          /* auto */
        CHECK_EQ(j["server"]["max_tokens_cap"].get<int64_t>(), 4096);
        const json& r = j["request_defaults"];
        CHECK_NEAR(r["presence_penalty"].get<double>(), 1.5, 1e-6);
        CHECK_EQ(r["chat_template_kwargs"]["enable_thinking"].get<bool>(), false);
        CHECK_EQ(r["chat_template_kwargs"]["tag"].get<std::string>(), std::string("x"));
    }
}

/* `preserve_thinking` at the top level is the template variable, like `enable_thinking`. Some
 * clients (oh-my-pi) send it there and in chat_template_kwargs on every request to a Qwen model,
 * so refusing it as an unknown field would fail all of them. */
TEST(preserve_thinking_at_the_top_level_reaches_the_template) {
    Fixture f;
    OaiLimits lim = f.limits();
    chat_with(f, lim, R"("preserve_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("true"));
    CHECK(f.tmpl.last_opt.enable_thinking);                /* a history knob, not a thinking one */

    /* oh-my-pi's exact pair, and the top-level spelling winning a disagreement. */
    chat_with(f, lim, R"("preserve_thinking":true,"chat_template_kwargs":{"preserve_thinking":true})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("true"));
    chat_with(f, lim, R"("preserve_thinking":false,"chat_template_kwargs":{"preserve_thinking":true})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("false"));

    chat_with(f, lim, "");
    CHECK(!f.tmpl.last_opt.template_kwargs.count("preserve_thinking"));

    OaiRequest r;
    ApiError e;
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"preserve_thinking":"yes"})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK_EQ(e.param, std::string("preserve_thinking"));
    /* A chat field, not a completion one: /v1/completions has no template to hand it to and
     * ignores it like any field it does not read. */
    CHECK_EQ(parse_completion_request(R"({"prompt":"ab","preserve_thinking":true})", f.odeps(), lim,
                                      &r, &e), RAD_OK);
}

/* parallel_tool_calls reaches the template because it is the one tool-choice knob the generated
 * grammar honours: it narrows the repetition rule rather than being advice. */
TEST(parallel_tool_calls_false_reaches_the_template) {
    Fixture f;
    f.sched.reply = "TOOL:a:{}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"a"}}],
                                "parallel_tool_calls":false})");
    CHECK_EQ(r.status, 200);
    CHECK(!f.tmpl.last_opt.parallel_tool_calls);
    CHECK_EQ(f.tmpl.last_n_tools, 1);
}

/* A forced choice reaches the template as "required", with the function it names; the grammar
 * that holds the reply to it is built from that (chatparse_test). */
TEST(a_forced_tool_choice_reaches_the_template) {
    Fixture f;
    f.sched.reply = "TOOL:a:{}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"a"}}],
                                "tool_choice":"required"})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_opt.tool_choice, std::string("required"));
    CHECK(f.tmpl.last_opt.tool_name.empty());

    HttpResponse r2 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                 "tools":[{"type":"function","function":{"name":"a"}},
                                          {"type":"function","function":{"name":"b"}}],
                                 "tool_choice":{"type":"function","function":{"name":"b"}}})");
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(f.tmpl.last_opt.tool_choice, std::string("required"));
    CHECK_EQ(f.tmpl.last_opt.tool_name, std::string("b"));
    CHECK_EQ(f.tmpl.last_n_tools, 2);
}

/* What cannot be forced is a 400 naming tool_choice: a call with no tools, a function the request
 * does not offer, and a call that would have to share the whole reply with a document. */
TEST(a_forced_tool_choice_that_cannot_be_kept_is_refused) {
    Fixture f;
    auto refused = [&](const std::string& body) {
        HttpResponse r = f.post("/v1/chat/completions", body);
        return r.status == 400 && r.body.find("tool_choice") != std::string::npos;
    };
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"})"));
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],
                      "tools":[{"type":"function","function":{"name":"a"}}],
                      "tool_choice":{"type":"function","function":{"name":"zz"}}})"));
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],
                      "tools":[{"type":"function","function":{"name":"a"}}],
                      "tool_choice":"required","response_format":{"type":"json_object"}})"));
}

/* A grammar the compiler cannot run is a 400 at admission, not a request that dies mid-stream. */
TEST(a_grammar_that_cannot_be_run_is_refused_at_the_door) {
    Fixture f;
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "grammar":"root ::= bad"})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("grammar") != std::string::npos);
}

/* The same on /v1/completions, which has no template in between: the grammar is the client's own
 * text, and the first place it can be refused is here. */
TEST(a_completion_grammar_that_cannot_be_run_is_refused_at_the_door) {
    Fixture f;
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","grammar":"root ::= bad"})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("grammar") != std::string::npos);
}

/* A model whose tool calls the reply parser cannot describe still chats, and a request that sends
 * tools is refused at admission with the reason, never answered with calls read by another
 * parser or with prose where a call was meant. */
TEST(tools_for_a_model_whose_calls_cannot_be_read_are_refused_with_the_reason) {
    Fixture f;
    f.tmpl.refusal = "the template writes each call as a JSON object";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("tools") != std::string::npos);
    CHECK(r.body.find("JSON object") != std::string::npos);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);
}

TEST(an_incomplete_tool_call_is_content_not_a_tool_call) {
    Fixture f;
    /* The model stopped mid-object. Reporting this as a tool call would make the client invoke
     * the tool with truncated arguments (json_partial.h). */
    f.sched.reply = "TOOL:get_weather:{\"city\":\"ber";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    const auto& c = j["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(!c["message"].contains("tool_calls"));
    CHECK(c["message"]["content"].get<std::string>().rfind("TOOL:", 0) == 0);
}

/* ================================================================== streaming */

/* Split an Anthropic event stream ("event: <name>\ndata: <json>\n\n" per event) into
 * (name, payload) pairs, setting `*failures_at_line` and stopping at the first event that breaks
 * the framing. */
struct AnthEvent { std::string name, payload; };
static std::vector<AnthEvent> anth_events(const std::string& body, int* failures_at_line) {
    std::vector<AnthEvent> out;
    size_t i = 0;
    while (i < body.size()) {
        size_t end = body.find("\n\n", i);
        if (end == std::string::npos) { *failures_at_line = __LINE__; break; }
        const std::string blk = body.substr(i, end - i);
        i = end + 2;
        if (blk.rfind("event: ", 0) != 0) { *failures_at_line = __LINE__; break; }
        const size_t nl = blk.find('\n');
        if (nl == std::string::npos || nl <= 7) { *failures_at_line = __LINE__; break; }
        if (blk.compare(nl + 1, 6, "data: ") != 0) { *failures_at_line = __LINE__; break; }
        out.push_back({ blk.substr(7, nl - 7), blk.substr(nl + 7) });
    }
    return out;
}

/* A chat stream sends no data event while the prompt is read: its first event is the opening
 * `role: assistant` chunk, written with the first output, as vLLM and llama-server write it. A
 * client that times its first data chunk (llama-benchy's time to first response) then measures
 * the time to the first token. A reply that produces nothing still opens before its final chunk. */
TEST(a_chat_streams_first_event_comes_with_the_first_token) {
    Fixture f(true, 0, "", true);
    const char* body = R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                           "logprobs":true,"stream_options":{"include_usage":true}})";
    f.sched.manual = true;
    HttpResponse r = f.post("/v1/chat/completions", body);
    f.sched.manual = false;
    REQUIRE(r.streaming());
    const size_t i = f.sched.seen.size() - 1;
    /* The stream is open and the prompt not yet read: nothing on the wire. */
    std::string c;
    for (int k = 0; k < 2; ++k) {
        c.clear();
        CHECK(r.next(c));
        CHECK(c.empty());
    }
    for (const char t : { 'h', 'e' }) {
        TokenLogprobs lp;
        lp.token = t;
        lp.logprob = -0.5f;
        f.sched.sink(i)->push(t, &lp);
    }
    c.clear();
    CHECK(r.next(c));
    int bad = 0;
    std::vector<std::string> ps = sse_payloads(c, &bad);
    CHECK_EQ(bad, 0);
    REQUIRE(ps.size() == 2);
    const json open = json::parse(ps[0])["choices"][0];
    CHECK(open["delta"] == json::parse(R"({"role":"assistant","content":""})"));
    CHECK(!open.contains("logprobs"));
    const json first = json::parse(ps[1])["choices"][0];
    CHECK_EQ(first["delta"].value("content", ""), std::string("he"));
    CHECK_EQ(first["logprobs"]["content"].size(), (size_t)2);
    f.sched.finish(i, Finish::Stop);
    const std::string rest = drain(r);
    CHECK(!has(rest, "\"role\""));

    /* No output at all: the opening chunk, then the final one. */
    f.sched.manual = true;
    HttpResponse e = f.post("/v1/chat/completions", body);
    f.sched.manual = false;
    REQUIRE(e.streaming());
    c.clear();
    CHECK(e.next(c));
    CHECK(c.empty());
    f.sched.finish(f.sched.seen.size() - 1, Finish::Length);
    ps = sse_payloads(drain(e), &bad);
    CHECK_EQ(bad, 0);
    REQUIRE(ps.size() >= 3);
    CHECK(json::parse(ps[0])["choices"][0]["delta"] ==
          json::parse(R"({"role":"assistant","content":""})"));
    CHECK_EQ(json::parse(ps[1])["choices"][0].value("finish_reason", ""), std::string("length"));

    /* A first token whose text is held back (it may begin a stop string) still opens the stream
     * when it is generated. */
    f.sched.manual = true;
    HttpResponse h = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                "stop":["xyz"]})");
    f.sched.manual = false;
    REQUIRE(h.streaming());
    const size_t k = f.sched.seen.size() - 1;
    c.clear();
    CHECK(h.next(c));
    CHECK(c.empty());
    f.sched.push(k, "x");
    c.clear();
    CHECK(h.next(c));
    ps = sse_payloads(c, &bad);
    REQUIRE(ps.size() == 1);
    CHECK(json::parse(ps[0])["choices"][0]["delta"] ==
          json::parse(R"({"role":"assistant","content":""})"));
    f.sched.push(k, "a");
    f.sched.finish(k, Finish::Stop);
    const std::string tail = drain(h);
    CHECK(has(tail, "\"content\":\"xa\""));
    CHECK(!has(tail, "\"role\""));
}

/* The Anthropic stream sends no event while the prompt is read either: message_start goes out
 * with the first token, followed by the first block. A request that fails before any token
 * still gets message_start ahead of its error. The Responses stream opens with
 * response.created and response.in_progress at once, as OpenAI's does: they report the response
 * object, not output. */
TEST(the_messages_streams_first_event_comes_with_the_first_token) {
    Fixture f;
    const char* body = R"({"model":"m","max_tokens":16,"stream":true,
                           "messages":[{"role":"user","content":"hi"}]})";
    f.sched.manual = true;
    HttpResponse r = f.post("/v1/messages", body);
    f.sched.manual = false;
    REQUIRE(r.streaming());
    const size_t i = f.sched.seen.size() - 1;
    std::string c;
    for (int k = 0; k < 2; ++k) {
        c.clear();
        CHECK(r.next(c));
        CHECK(c.empty());
    }
    f.sched.push(i, "he");
    c.clear();
    CHECK(r.next(c));
    int bad = 0;
    std::vector<AnthEvent> evs = anth_events(c, &bad);
    CHECK_EQ(bad, 0);
    REQUIRE(evs.size() >= 3);
    CHECK_EQ(evs[0].name, std::string("message_start"));
    CHECK_EQ(evs[1].name, std::string("content_block_start"));
    CHECK_EQ(evs[2].name, std::string("content_block_delta"));
    f.sched.finish(i, Finish::Stop);
    const std::string rest = drain(r);
    CHECK(!has(rest, "message_start"));
    CHECK(has(rest, "message_stop"));

    f.sched.manual = true;
    HttpResponse e = f.post("/v1/messages", body);
    f.sched.manual = false;
    REQUIRE(e.streaming());
    c.clear();
    CHECK(e.next(c));
    CHECK(c.empty());
    sink_fail(f.sched.sink(f.sched.seen.size() - 1), RAD_E_IO, "the step failed");
    evs = anth_events(drain(e), &bad);
    REQUIRE(evs.size() == 2);
    CHECK_EQ(evs[0].name, std::string("message_start"));
    CHECK_EQ(evs[1].name, std::string("error"));

    f.sched.manual = true;
    HttpResponse o = f.post("/v1/responses", R"({"model":"m","input":"hi","stream":true})");
    f.sched.manual = false;
    REQUIRE(o.streaming());
    c.clear();
    CHECK(o.next(c));
    CHECK(has(c, "\"response.created\"") && has(c, "\"response.in_progress\""));
    f.sched.finish(f.sched.seen.size() - 1, Finish::Stop);
    drain(o);
}

TEST(streaming_chat_is_valid_sse_and_ends_with_done) {
    Fixture f;
    f.sched.reply = "hey";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "stream":true,
                                "stream_options":{"include_usage":true}})");
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());
    CHECK(r.content_type.rfind("text/event-stream", 0) == 0);

    std::string body = drain(r);
    /* Every event is "data: <payload>\n\n" and the last one is the sentinel. */
    CHECK(body.size() > sizeof(SSE_DONE));
    CHECK(body.compare(body.size() - std::strlen(SSE_DONE), std::strlen(SSE_DONE), SSE_DONE) == 0);

    int bad = 0;
    auto payloads = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);
    CHECK(payloads.size() >= 3);
    CHECK_EQ(payloads.back(), std::string("[DONE]"));

    std::string text;
    bool saw_role = false, saw_finish = false, saw_usage = false;
    for (size_t i = 0; i + 1 < payloads.size(); ++i) {
        json j = json::parse(payloads[i]);
        CHECK_EQ(j["object"].get<std::string>(), std::string("chat.completion.chunk"));
        CHECK(j["id"].get<std::string>().rfind("chatcmpl-", 0) == 0);
        if (j.contains("usage")) { saw_usage = true; CHECK_EQ((int)j["choices"].size(), 0); continue; }
        const auto& c = j["choices"][0];
        CHECK_EQ(c["index"].get<int>(), 0);
        if (c["delta"].contains("role")) saw_role = true;
        if (c["delta"].contains("content")) text += c["delta"]["content"].get<std::string>();
        if (!c["finish_reason"].is_null()) {
            saw_finish = true;
            CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
            CHECK_EQ((int)c["delta"].size(), 0);
        }
    }
    CHECK(saw_role);
    CHECK(saw_finish);
    CHECK(saw_usage);
    CHECK_EQ(text, std::string("hey"));
}

TEST(streaming_completions_use_the_text_completion_chunk_shape) {
    Fixture f;
    f.sched.reply = "pq";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","stream":true})");
    CHECK_EQ(r.status, 200);
    std::string body = drain(r);
    int bad = 0;
    auto p = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);
    CHECK_EQ(p.back(), std::string("[DONE]"));
    std::string text;
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        json j = json::parse(p[i]);
        CHECK_EQ(j["object"].get<std::string>(), std::string("text_completion"));
        text += j["choices"][0]["text"].get<std::string>();
    }
    CHECK_EQ(text, std::string("pq"));
}

/* A request the engine failed says what failed, in the 500 and in a stream's error event: the
 * reason and the status the engine stopped on, never a generic "ok". */
TEST(a_request_the_engine_failed_says_what_failed) {
    Fixture f;
    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_UNSUPPORTED;
    f.sched.fail_why = "the engine stopped on an error and is not serving";
    const std::string want = "the engine failed this request: the engine stopped on an error and "
                             "is not serving (unsupported)";

    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 500);
    CHECK_EQ(json::parse(r.body)["error"]["message"].get<std::string>(), want);

    HttpResponse s = f.post("/v1/completions", R"({"prompt":"hi","stream":true})");
    CHECK_EQ(s.status, 200);
    int bad = 0, errors = 0;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        json j = json::parse(p);
        if (!j.contains("error")) continue;
        ++errors;
        CHECK_EQ(j["error"]["message"].get<std::string>(), want);
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(errors, 1);

    /* With no reason given, the status alone, and never "ok". */
    f.sched.fail_why = nullptr;
    HttpResponse t = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(json::parse(t.body)["error"]["message"].get<std::string>(),
             std::string("the engine failed this request: unsupported"));
}

/* A reply that ends inside its last call, read by the real parser in Qwen's format: that call is
 * the text the model wrote, after the content before it, and nothing else moves (reasoning stays
 * reasoning, a call that closed before it stays a call, and the finish reason is "stop", streamed
 * or not). The stream has already sent the open call's deltas and cannot take them back; its
 * finish reason, the buffered one, is what says not to run it. */
TEST(a_reply_ending_inside_a_call_keeps_its_reasoning_and_earlier_calls) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"bash",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},
                      "required":["command"]}}}])");
    const std::string call1 = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                              "</function>\n</tool_call>\n";
    const std::string open = "<tool_call>\n<function=bash>\n<parameter=command>\ncat /a";
    const std::string reply = "Look first.\n</think>\n\nChecking.\n" + call1 + open;
    const std::string body = R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1000,
        "tools":)" + tools.dump();

    Fixture f;
    f.tmpl.real_parser = qwen_reply_parser(tools);
    REQUIRE(f.tmpl.real_parser != nullptr);
    f.sched.reply = reply;
    HttpResponse r = f.post("/v1/chat/completions", body + "}");
    CHECK_EQ(r.status, 200);
    const json c = json::parse(r.body)["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK_EQ(c["message"]["reasoning_content"].get<std::string>(), std::string("Look first.\n"));
    CHECK_EQ(c["message"]["content"].get<std::string>(), "Checking.\n" + open);
    REQUIRE(c["message"].contains("tool_calls") && c["message"]["tool_calls"].size() == 1);
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["arguments"].get<std::string>(),
             std::string("{\"command\":\"ls\"}"));

    /* Streamed: the same reasoning, the closed call, the same finish reason. */
    HttpResponse s = f.post("/v1/chat/completions", body + R"(,"stream":true})");
    CHECK_EQ(s.status, 200);
    int bad = 0;
    std::string reasoning, finish;
    std::vector<std::string> args;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        if (j["choices"].empty()) continue;
        const json& ch = j["choices"][0];
        if (!ch["finish_reason"].is_null()) finish = ch["finish_reason"].get<std::string>();
        const json& d = ch["delta"];
        if (d.contains("reasoning_content")) reasoning += d["reasoning_content"].get<std::string>();
        if (!d.contains("tool_calls")) continue;
        const size_t k = d["tool_calls"][0]["index"].get<size_t>();
        if (k >= args.size()) args.resize(k + 1);
        if (d["tool_calls"][0]["function"].contains("arguments"))
            args[k] += d["tool_calls"][0]["function"]["arguments"].get<std::string>();
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(finish, std::string("stop"));
    CHECK_EQ(reasoning, std::string("Look first.\n"));
    REQUIRE(!args.empty());
    CHECK_EQ(args[0], std::string("{\"command\":\"ls\"}"));
}

TEST(a_streamed_tool_call_is_named_before_its_arguments_flow) {
    Fixture f;
    f.sched.reply = "TOOL:lookup:{\"q\":\"x\"}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                "tools":[{"type":"function","function":{"name":"lookup"}}]})");
    CHECK_EQ(r.status, 200);
    std::string body = drain(r);
    int bad = 0;
    auto p = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);

    bool named = false;
    std::string args;
    std::string finish;
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        json j = json::parse(p[i]);
        if (j["choices"].empty()) continue;
        const auto& d = j["choices"][0]["delta"];
        if (!j["choices"][0]["finish_reason"].is_null())
            finish = j["choices"][0]["finish_reason"].get<std::string>();
        if (!d.contains("tool_calls")) continue;
        const auto& tc = d["tool_calls"][0];
        if (tc.contains("function") && tc["function"].contains("name")) {
            /* The name arrives with the entry, before any arguments; an entry with no name is an
             * index the client can never resolve. */
            CHECK(!named);
            CHECK(args.empty());
            CHECK_EQ(tc["function"]["name"].get<std::string>(), std::string("lookup"));
            CHECK(tc.contains("id"));
            CHECK_EQ(tc["type"].get<std::string>(), std::string("function"));
            named = true;
        }
        if (tc.contains("function") && tc["function"].contains("arguments"))
            args += tc["function"]["arguments"].get<std::string>();
    }
    CHECK(named);
    CHECK_EQ(args, std::string("{\"q\":\"x\"}"));
    CHECK_EQ(finish, std::string("tool_calls"));
}

/* A finished completion streams its whole text. The hold-back rules wait for bytes that may still
 * arrive; at the end none will, so a text ending in `\` or a partial `\uXXXX` (ordinary text in a
 * completion) must still arrive whole. With and without a stop string, since a stop list adds its
 * own hold-back. */
TEST(a_streamed_completion_ends_with_its_whole_text) {
    for (const char* reply : {"ab\\", "x\\u12", "q\\u", "tail\\\\"}) {
        for (const char* body : {R"({"prompt":"hi","stream":true})",
                                 R"({"prompt":"hi","stream":true,"stop":["ZZZZ"]})"}) {
            Fixture f;
            f.sched.reply = reply;
            HttpResponse r = f.post("/v1/completions", body);
            CHECK_EQ(r.status, 200);
            int bad = 0;
            auto p = sse_payloads(drain(r), &bad);
            CHECK_EQ(bad, 0);
            std::string text;
            for (size_t i = 0; i + 1 < p.size(); ++i) {
                json j = json::parse(p[i]);
                if (!j["choices"].empty()) text += j["choices"][0]["text"].get<std::string>();
            }
            CHECK_EQ(text, std::string(reply));
        }
    }
}

/* A request that sent no tools is never told `tool_calls`, streamed or not: it has nothing to
 * dispatch a call to. The fake template hands back a parser for every render, as a real one does
 * for the reasoning split, and for a request without tools that parser reads the call syntax as
 * text, so the reply comes back as the text it is on both paths. */
TEST(a_chat_stream_that_sent_no_tools_never_reports_a_tool_call) {
    const std::string reply = "TOOL:lookup:{\"q\":\"x\"}";
    {
        Fixture f;
        f.sched.reply = reply;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"stream":true})");
        CHECK_EQ(r.status, 200);
        int bad = 0;
        auto p = sse_payloads(drain(r), &bad);
        CHECK_EQ(bad, 0);
        std::string content, finish;
        bool saw_call = false;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            json j = json::parse(p[i]);
            if (j["choices"].empty()) continue;
            const auto& c = j["choices"][0];
            if (c["delta"].contains("tool_calls")) saw_call = true;
            if (c["delta"].contains("content")) content += c["delta"]["content"].get<std::string>();
            if (!c["finish_reason"].is_null()) finish = c["finish_reason"].get<std::string>();
        }
        CHECK(!saw_call);
        CHECK_EQ(finish, std::string("stop"));
        CHECK_EQ(content, reply);
    }
    {
        Fixture f;
        f.sched.reply = reply;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        json j = json::parse(r.body);
        const auto& c = j["choices"][0];
        CHECK(!c["message"].contains("tool_calls"));
        CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
        CHECK_EQ(c["message"]["content"].get<std::string>(), reply);
    }
}

/* One usage chunk per stream, after every choice's final chunk and right before [DONE]. `n`
 * choices share one prompt and one total; a chunk per choice would count the prompt n times and
 * land between choices. Both endpoints. */
TEST(n_streamed_choices_share_one_usage_chunk_after_every_final_chunk) {
    const char* bodies[] = {
        R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"n":3,
            "stream_options":{"include_usage":true}})",
        R"({"prompt":"hi","stream":true,"n":3,"stream_options":{"include_usage":true}})",
    };
    const char* paths[] = {"/v1/chat/completions", "/v1/completions"};
    for (int k = 0; k < 2; ++k) {
        Fixture f;
        f.sched.reply = "ok";
        HttpResponse r = f.post(paths[k], bodies[k]);
        CHECK_EQ(r.status, 200);
        int bad = 0;
        auto p = sse_payloads(drain(r), &bad);
        CHECK_EQ(bad, 0);
        REQUIRE(p.size() >= 2);
        CHECK_EQ(p.back(), std::string("[DONE]"));

        int usage = 0, finals = 0;
        size_t usage_at = 0, last_final_at = 0;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            json j = json::parse(p[i]);
            if (j.contains("usage")) {
                ++usage;
                usage_at = i;
                CHECK_EQ((int)j["choices"].size(), 0);
                CHECK_EQ(j["usage"]["completion_tokens"].get<int64_t>(), (int64_t)6);
                continue;
            }
            if (!j["choices"][0]["finish_reason"].is_null()) { ++finals; last_final_at = i; }
        }
        CHECK_EQ(usage, 1);
        CHECK_EQ(finals, 3);
        CHECK(usage_at > last_final_at);
        CHECK_EQ(usage_at, p.size() - 2);
    }
}

/* ================================================================== reasoning */

/* How much of completion_tokens was reasoning, on both paths: the reply opens inside the block
 * (the render says so), and `|` (one token to FakeTokenizer) closes it after three body tokens,
 * so four of the seven tokens are reasoning, the close included. A model whose block cannot be
 * counted reports no breakdown rather than a zero. */
TEST(usage_reports_the_reasoning_tokens_inside_completion_tokens) {
    {
        Fixture f;
        f.tmpl.think_close = "|";
        f.tmpl.think_starts_open = true;
        f.sched.reply = "abc|xyz";
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        CHECK_EQ(j["usage"]["completion_tokens"].get<int>(), 7);
        CHECK_EQ(j["usage"]["completion_tokens_details"]["reasoning_tokens"].get<int>(), 4);

        HttpResponse st = f.post("/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                     "stream_options":{"include_usage":true}})");
        int bad = 0;
        auto p = sse_payloads(drain(st), &bad);
        CHECK_EQ(bad, 0);
        REQUIRE(p.size() >= 2);
        const json u = json::parse(p[p.size() - 2]);
        CHECK_EQ(u["usage"]["completion_tokens_details"]["reasoning_tokens"].get<int>(), 4);
    }
    {
        Fixture f;
        f.tmpl.think_close = "</t>";       /* four tokens: not a countable block */
        f.tmpl.think_starts_open = true;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        CHECK(!json::parse(r.body)["usage"].contains("completion_tokens_details"));
    }
}

/* A reply that starts inside its reasoning block keeps its stop strings for the answer: one written
 * while thinking is text, and the first one after the closer cuts. The scheduler, which cannot
 * tell reasoning from answer, is handed none. A multi-token closer arrives across drains, which
 * the incremental search must bridge. */
TEST(stop_strings_wait_for_the_reasoning_closer) {
    for (const char* close : { "|", "</t>" }) {
        Fixture f;
        f.tmpl.think_close = close;
        f.tmpl.think_starts_open = true;
        f.sched.reply = std::string("aENDb") + close + "cENDd";
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(),
                 std::string("aENDb") + close + "c");
        CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
        CHECK(f.sched.last_stop.empty());
    }
    /* A stop string only inside the reasoning never stops the reply. */
    Fixture f;
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    f.sched.reply = "aENDb|cd";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
    CHECK_EQ(r.status, 200);
    const json j = json::parse(r.body);
    CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(), std::string("aENDb|cd"));
    CHECK(!f.sched.saw_cancel());             /* ended by the reply itself, not cut */
}

/* The budget reaches the engine in every spelling (vLLM's, Qwen's and llama.cpp's two) with the
 * block it is held on; one the block cannot hold, or two spellings that disagree, are 400s naming
 * the field, and a negative one is llama.cpp's "no budget". */
TEST(a_thinking_budget_reaches_the_engine_or_is_refused_by_name) {
    Fixture f;
    f.tmpl.think_open = "{";
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    for (const char* k : { "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
                           "reasoning_budget_tokens" }) {
        HttpResponse r = f.post("/v1/chat/completions",
                                std::string(R"({"messages":[{"role":"user","content":"hi"}],")") +
                                    k + R"(":7})");
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.thinking_budget, 7);
        CHECK_EQ(f.sched.last_sp.reasoning.close, (int32_t)'|');
        CHECK_EQ(f.sched.last_sp.reasoning.open, (int32_t)'{');
        CHECK(f.sched.last_sp.reasoning.starts_open);
    }

    struct Case { const char* extra; const char* param; };
    for (const Case& c : { Case{ R"("thinking_token_budget":3,"thinking_budget":4)",
                                 "thinking_budget" },
                           Case{ R"("thinking_budget":"3")", "thinking_budget" } }) {
        HttpResponse r = f.post("/v1/chat/completions",
                                std::string(R"({"messages":[{"role":"user","content":"hi"}],)") +
                                    c.extra + "}");
        CHECK_EQ(r.status, 400);
        CHECK_EQ(json::parse(r.body)["error"]["param"].get<std::string>(), std::string(c.param));
    }

    {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],
                                    "reasoning_budget_tokens":-1})");
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.thinking_budget, -1);
    }

    Fixture multi;
    multi.tmpl.think_close = "</t>";
    multi.tmpl.think_starts_open = true;
    HttpResponse r = multi.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],
                                    "thinking_token_budget":5})");
    CHECK_EQ(r.status, 400);
    CHECK(json::parse(r.body)["error"]["message"].get<std::string>().find("4 tokens") !=
          std::string::npos);

    Fixture none;
    HttpResponse n = none.post("/v1/chat/completions",
                               R"({"messages":[{"role":"user","content":"hi"}],
                                   "thinking_token_budget":5})");
    CHECK_EQ(n.status, 400);
    CHECK(json::parse(n.body)["error"]["message"].get<std::string>().find("no reasoning block") !=
          std::string::npos);
}

/* The deployment's budget applies to a request that names none, loses to one that does, and is
 * not applied where the block cannot be counted (it could not be held there). The startup check
 * says why a model cannot take it at all. */
TEST(the_deployment_s_thinking_budget_is_a_default_a_request_can_override) {
    Fixture f;
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    OaiLimits lim = f.limits();
    lim.default_thinking_budget = 64;
    CHECK_EQ(chat_with(f, lim, "").sp.thinking_budget, 64);
    CHECK_EQ(chat_with(f, lim, R"("thinking_token_budget":3)").sp.thinking_budget, 3);
    CHECK_EQ(chat_with(f, lim, R"("reasoning_budget_tokens":-1)").sp.thinking_budget, -1);
    CHECK_EQ(chat_with(f, f.limits(), "").sp.thinking_budget, -1);

    f.tmpl.think_close.clear();
    CHECK_EQ(chat_with(f, lim, "").sp.thinking_budget, -1);

    Fixture g;
    ServerOptions o;
    o.model_id = "test-model";
    o.default_max_tokens = 32;
    o.default_thinking_budget = 64;
    Deps d;
    d.sched = &g.sched;
    d.tok = &g.tok;
    d.chat = &g.tmpl;
    d.grammar = &g.gram;
    Server s(d, o);
    CHECK(s.check_thinking_budget().find("no reasoning block") != std::string::npos);
    g.tmpl.think_close = "|";
    g.tmpl.think_starts_open = true;
    CHECK(s.check_thinking_budget().empty());
}

/* ================================================================== timings */

/* Where the request's time went, in llama.cpp's shape, on every path a client can take: the
 * buffered body, each streamed choice's final chunk whether or not a usage chunk was requested,
 * and the usage chunk when it was. The fake places 8 prompt tokens with 4 from the cache, and
 * every token after the first is a verify step that proposed 3 drafts and took 1. */
TEST(a_response_carries_its_requests_timings_and_draft_counts) {
    auto check = [](const json& t, int64_t predicted, bool drafts) {
        CHECK_EQ(t["prompt_n"].get<int64_t>(), (int64_t)4);
        CHECK_EQ(t["cache_n"].get<int64_t>(), (int64_t)4);
        CHECK_EQ(t["predicted_n"].get<int64_t>(), predicted);
        CHECK(t["prompt_ms"].is_number());
        CHECK(t["predicted_ms"].is_number());
        CHECK(t["prompt_ms"].get<double>() >= 0.0);
        for (const char* k : { "prompt_per_token_ms", "prompt_per_second",
                               "predicted_per_token_ms", "predicted_per_second" })
            CHECK(t[k].is_number() || t[k].is_null());
        CHECK_EQ(t.contains("draft_n"), drafts);
        if (drafts) {
            CHECK_EQ(t["draft_n"].get<int64_t>(), (predicted - 1) * 3);
            CHECK_EQ(t["draft_n_accepted"].get<int64_t>(), predicted - 1);
        }
    };
    {
        Fixture f;
        f.sched.drafts = 3;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        REQUIRE(j.contains("timings"));
        check(j["timings"], 5, true);
        CHECK_EQ(j["timings"]["predicted_n"].get<int64_t>(),
                 j["usage"]["completion_tokens"].get<int64_t>());
    }
    {
        Fixture f;
        HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","max_tokens":8})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        REQUIRE(j.contains("timings"));
        check(j["timings"], 5, false);
    }
    const char* paths[] = { "/v1/chat/completions", "/v1/completions" };
    const char* bodies[] = { R"({"messages":[{"role":"user","content":"hi"}],"stream":true)",
                             R"({"prompt":"hi","stream":true)" };
    for (int k = 0; k < 2; ++k)
        for (bool usage : { false, true }) {
            Fixture f;
            f.sched.drafts = 3;
            HttpResponse r = f.post(paths[k], std::string(bodies[k]) +
                                    (usage ? R"(,"stream_options":{"include_usage":true}})"
                                           : "}"));
            CHECK_EQ(r.status, 200);
            int bad = 0;
            auto p = sse_payloads(drain(r), &bad);
            CHECK_EQ(bad, 0);
            int finals = 0, with_timings = 0;
            for (size_t i = 0; i + 1 < p.size(); ++i) {
                const json j = json::parse(p[i]);
                if (j.contains("usage")) {
                    REQUIRE(j.contains("timings"));
                    check(j["timings"], 5, true);
                    ++with_timings;
                    continue;
                }
                if (j["choices"][0]["finish_reason"].is_null()) {
                    CHECK(!j.contains("timings"));
                    continue;
                }
                ++finals;
                REQUIRE(j.contains("timings"));
                check(j["timings"], 5, true);
                ++with_timings;
            }
            CHECK_EQ(finals, 1);
            CHECK_EQ(with_timings, usage ? 2 : 1);
        }
}

/* The rates are tokens over the time that produced them, the first token counting as the prompt
 * phase's; a rate whose count or time is zero is null, as is a prompt time the engine never
 * reported. */
TEST(timings_rates_divide_the_tokens_by_the_time_that_produced_them) {
    TimingsOut t;
    t.valid = true;
    t.prompt_n = 100;
    t.prompt_ms = 50.0;
    t.predicted_n = 10;
    t.decode_n = 9;
    t.predicted_ms = 90.0;
    json j = timings_json(t);
    CHECK_NEAR(j["prompt_per_second"].get<double>(), 2000.0, 1e-9);
    CHECK_NEAR(j["prompt_per_token_ms"].get<double>(), 0.5, 1e-12);
    CHECK_NEAR(j["predicted_per_second"].get<double>(), 100.0, 1e-9);
    CHECK_NEAR(j["predicted_per_token_ms"].get<double>(), 10.0, 1e-12);
    CHECK(!j.contains("draft_n"));

    t.decode_n = 0;
    t.predicted_ms = 0.0;
    t.prompt_ms = -1.0;
    j = timings_json(t);
    CHECK(j["predicted_per_second"].is_null());
    CHECK(j["predicted_per_token_ms"].is_null());
    CHECK(j["prompt_ms"].is_null());
    CHECK(j["prompt_per_second"].is_null());
}

/* The server's own time, from when the body had arrived: ttft_ms covers everything up to the first
 * token, total_ms up to the last, on the buffered body and a stream's final chunk. A request made
 * in-process has no arrival to count from and reports neither. */
TEST(timings_count_ttft_and_total_from_the_bodys_arrival) {
    auto arrived = [](const char* path, const std::string& body) {
        HttpRequest q;
        q.method = "POST";
        q.path = path;
        q.body = body;
        /* Five milliseconds before now, on the steady clock the sinks count on. */
        q.t_received_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count() - 5000000;
        return q;
    };
    auto check = [](const json& t) {
        REQUIRE(t["ttft_ms"].is_number());
        REQUIRE(t["total_ms"].is_number());
        CHECK(t["ttft_ms"].get<double>() >= 5.0);
        CHECK(t["ttft_ms"].get<double>() >= t["prompt_ms"].get<double>());
        CHECK_NEAR(t["total_ms"].get<double>(),
                   t["ttft_ms"].get<double>() + t["predicted_ms"].get<double>(), 1e-6);
    };
    {
        Fixture f;
        HttpResponse r = f.srv->router().dispatch(
            arrived("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})"));
        CHECK_EQ(r.status, 200);
        check(json::parse(r.body)["timings"]);
    }
    {
        Fixture f;
        HttpResponse r = f.srv->router().dispatch(
            arrived("/v1/completions", R"({"prompt":"hi","stream":true})"));
        CHECK_EQ(r.status, 200);
        int bad = 0, finals = 0;
        auto p = sse_payloads(drain(r), &bad);
        CHECK_EQ(bad, 0);
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            const json j = json::parse(p[i]);
            if (j["choices"][0]["finish_reason"].is_null()) continue;
            ++finals;
            check(j["timings"]);
        }
        CHECK_EQ(finals, 1);
    }
    {
        Fixture f;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        const json t = json::parse(r.body)["timings"];
        CHECK(t["ttft_ms"].is_null());
        CHECK(t["total_ms"].is_null());
    }
}

/* The sink keeps the first admission across a preemption and sums the drafts it is told of. */
TEST(a_sink_keeps_its_first_admission_and_sums_its_drafts) {
    Sink s;
    CHECK(s.t_admitted_ns() < 0);
    s.mark_admitted();
    const int64_t a = s.t_admitted_ns();
    CHECK(a >= 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    s.mark_admitted();
    CHECK_EQ(s.t_admitted_ns(), a);
    s.add_drafts(4, 2);
    s.add_drafts(3, 3);
    CHECK_EQ(s.n_drafted(), (int64_t)7);
    CHECK_EQ(s.n_draft_accepted(), (int64_t)5);
}

/* A request the scheduler refuses gets id 0 and a failed sink, so the HTTP thread waiting on the
 * sink answers instead of hanging. An empty prompt is refused before the scheduler reads any
 * state, so an uninitialised one serves. */
TEST(a_refused_request_fails_its_sink) {
    Scheduler sch;
    SchedulerBridge br(&sch);
    Sink a, b, c;
    std::vector<Request> rs(2);
    rs[0].sink = &a;
    rs[1].sink = &b;
    std::vector<uint64_t> ids;
    br.submit_all(rs, &ids);
    CHECK_EQ(ids.size(), (size_t)2);
    for (int i = 0; i < 2; ++i) CHECK_EQ(ids[(size_t)i], (uint64_t)0);
    for (Sink* s : {&a, &b}) {
        CHECK(s->finish_reason() == Finish::Error);
        CHECK_EQ(s->status(), RAD_E_INVAL);
    }

    Request one;
    one.sink = &c;
    CHECK_EQ(br.submit(std::move(one)), (uint64_t)0);
    CHECK(c.finish_reason() == Finish::Error);
}

/* ================================================================== cancellation */

TEST(a_dropped_stream_cancels_at_the_scheduler) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;          /* never finishes on its own */

    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/chat/completions";
    q.body = R"({"messages":[{"role":"user","content":"hi"}],"stream":true})";
    std::atomic<bool> gone{false};
    q.closed = [&gone] { return gone.load(); };

    HttpResponse r = f.srv->router().dispatch(q);
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());

    /* Pull a few chunks so the generation is running. */
    for (int i = 0; i < 4; ++i) { std::string c; r.next(c); }
    CHECK(!f.sched.saw_cancel());
    CHECK_EQ(f.srv->in_flight(), 1LL);

    /* The peer goes away. The transport notices on its next pull and tears the provider down. */
    gone.store(true);
    std::string c;
    bool more = r.next(c);
    CHECK(!more);
    r.on_close(false);

    CHECK(f.sched.saw_cancel());
    CHECK_EQ(f.sched.cancelled[0], f.sched.submitted[0]);
    /* The admission slot is returned at the same moment, not when the generation would have
     * ended; otherwise an abandoned stream keeps the queue full for its whole max_tokens. */
    CHECK_EQ(f.srv->in_flight(), 0LL);

    /* The cancel must stop the worker, not merely be recorded; otherwise this join hangs for the
     * fake's full twenty-second hold. */
    f.sched.join_all();
}

TEST(a_dropped_non_streaming_request_cancels_too) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;

    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/chat/completions";
    q.body = R"({"messages":[{"role":"user","content":"hi"}]})";
    std::atomic<bool> gone{false};
    q.closed = [&gone] { return gone.load(); };

    /* The handler blocks until the answer or the drop, so the drop has to come from elsewhere. */
    std::thread hangup([&gone] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        gone.store(true);
    });
    HttpResponse r = f.srv->router().dispatch(q);
    hangup.join();

    CHECK_EQ(r.status, 499);
    CHECK(f.sched.saw_cancel());
    CHECK_EQ(f.srv->in_flight(), 0LL);
}

TEST(a_full_admission_queue_is_429_with_retry_after) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;

    /* queue_depth defaults to 8 * max_seqs, and the fixture set max_seqs to 4. */
    std::vector<HttpResponse> live;
    std::atomic<bool> never{false};
    int rejected = 0;
    for (int i = 0; i < 40; ++i) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = R"({"messages":[{"role":"user","content":"hi"}],"stream":true})";
        q.closed = [&never] { return never.load(); };
        HttpResponse r = f.srv->router().dispatch(q);
        if (r.status == 429) {
            ++rejected;
            bool has_retry = false;
            for (auto& h : r.headers) if (h.first == "Retry-After") has_retry = true;
            CHECK(has_retry);
            CHECK(r.body.find("queue") != std::string::npos);
        } else {
            CHECK_EQ(r.status, 200);
            live.push_back(std::move(r));
        }
    }
    CHECK(rejected > 0);
    CHECK_EQ((int)live.size(), 32);

    /* Unwind: every admitted stream is torn down, which must return every admission slot. */
    for (auto& r : live) if (r.on_close) r.on_close(false);
    CHECK_EQ(f.srv->in_flight(), 0LL);
    f.sched.join_all();
}

/* ================================================================== /metrics */

/* A minimal Prometheus text-format parser, enough to prove the output is well formed: every
 * non-comment line must be `name{labels} value` with a numeric value, and every family must
 * declare its type before its samples. */
struct PromDoc {
    std::map<std::string, std::string> type;      /* family -> gauge|counter|histogram */
    std::map<std::string, std::string> help;
    std::multimap<std::string, double> samples;   /* full sample name -> value */
    std::vector<std::string> errors;
};

static PromDoc parse_prom(const std::string& body) {
    PromDoc d;
    size_t i = 0;
    while (i < body.size()) {
        size_t nl = body.find('\n', i);
        if (nl == std::string::npos) nl = body.size();
        std::string line = body.substr(i, nl - i);
        i = nl + 1;
        if (line.empty()) continue;
        if (line[0] == '#') {
            size_t s1 = line.find(' ');
            size_t s2 = line.find(' ', s1 + 1);
            size_t s3 = line.find(' ', s2 + 1);
            std::string kind = line.substr(s1 + 1, s2 - s1 - 1);
            std::string name = line.substr(s2 + 1, s3 - s2 - 1);
            if (kind == "TYPE") d.type[name] = line.substr(s3 + 1);
            else if (kind == "HELP") d.help[name] = line.substr(s3 + 1);
            else d.errors.push_back("unknown comment: " + line);
            continue;
        }
        size_t sp = line.rfind(' ');
        if (sp == std::string::npos) { d.errors.push_back("no value: " + line); continue; }
        std::string name = line.substr(0, sp);
        std::string val = line.substr(sp + 1);
        try {
            d.samples.emplace(name, std::stod(val));
        } catch (...) {
            d.errors.push_back("bad value: " + line);
            continue;
        }
        size_t brace = name.find('{');
        std::string family = brace == std::string::npos ? name : name.substr(0, brace);
        if (brace != std::string::npos && name.back() != '}')
            d.errors.push_back("unclosed labels: " + line);
        /* Bucket, sum and count samples belong to the histogram family. */
        for (const char* suf : {"_bucket", "_sum", "_count"}) {
            size_t n = std::strlen(suf);
            if (family.size() > n && family.compare(family.size() - n, n, suf) == 0) {
                std::string base = family.substr(0, family.size() - n);
                if (d.type.count(base)) { family = base; break; }
            }
        }
        if (!d.type.count(family)) d.errors.push_back("sample before TYPE: " + family);
    }
    return d;
}

TEST(metrics_is_valid_prometheus_text_with_vllm_names) {
    Fixture f;
    f.sched.reply = "hello";
    f.sched.m.running = 3;
    f.sched.m.waiting = 7;
    f.sched.m.kv_util = 0.42;
    f.sched.m.prefix_hit_rate = 0.25;
    f.sched.m.decode_tps = 180.5;
    f.sched.m.draft_acceptance = 0.61;
    f.sched.m.preemptions = 2;

    /* One completed request so the histograms have an observation in them. */
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);

    HttpResponse r = f.get("/metrics");
    CHECK_EQ(r.status, 200);
    CHECK(r.content_type.find("version=0.0.4") != std::string::npos);

    PromDoc d = parse_prom(r.body);
    if (!d.errors.empty()) {
        for (auto& e : d.errors) fprintf(stderr, "    prom: %s\n", e.c_str());
    }
    CHECK_EQ((int)d.errors.size(), 0);

    /* The names existing dashboards query, with the types they expect. */
    CHECK_EQ(d.type["vllm:num_requests_running"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:num_requests_waiting"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:gpu_cache_usage_perc"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:prompt_tokens_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:generation_tokens_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:num_preemptions_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:request_success_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:time_to_first_token_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:time_per_output_token_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:e2e_request_latency_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:request_prompt_tokens"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:request_generation_tokens"], std::string("histogram"));

    /* Radiance's own metrics are under our own prefix, never invented under vllm:. */
    CHECK_EQ(d.type["radiance:draft_acceptance_ratio"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:decode_tokens_per_second"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:http_admission_rejected_total"], std::string("counter"));

    /* The scheduler's numbers reached the gauges. */
    auto find_one = [&](const std::string& prefix) -> double {
        for (auto& kv : d.samples)
            if (kv.first.rfind(prefix, 0) == 0) return kv.second;
        return -12345.0;
    };
    CHECK_NEAR(find_one("vllm:num_requests_running{"), 3.0, 1e-9);
    CHECK_NEAR(find_one("vllm:num_requests_waiting{"), 7.0, 1e-9);
    CHECK_NEAR(find_one("vllm:gpu_cache_usage_perc{"), 0.42, 1e-9);
    CHECK_NEAR(find_one("radiance:draft_acceptance_ratio{"), 0.61, 1e-9);
    CHECK_NEAR(find_one("vllm:generation_tokens_total{"), 5.0, 1e-9);

    /* An unmeasured decode floor is ABSENT, not 0: a floor of 0 seconds per token is an
     * infinitely fast one, and a scrape would plot it as a record. This fixture never completes a
     * window of 32 single-stream steps, so neither series may appear. */
    CHECK(r.body.find("radiance:decode_step_seconds_best") == std::string::npos);
    CHECK(r.body.find("radiance:decode_step_seconds_recent") == std::string::npos);

    /* Every sample carries the model_name label vLLM's dashboards group by. */
    for (auto& kv : d.samples)
        CHECK(kv.first.find("model_name=\"test-model\"") != std::string::npos);

    /* Histogram invariants: buckets are cumulative and non-decreasing, +Inf equals _count. */
    double last = -1.0, inf = -1.0, count = -1.0;
    for (auto& kv : d.samples) {
        if (kv.first.rfind("vllm:time_to_first_token_seconds_bucket", 0) == 0) {
            if (kv.first.find("le=\"+Inf\"") != std::string::npos) inf = kv.second;
            else { CHECK(kv.second >= last); last = kv.second; }
        }
        if (kv.first.rfind("vllm:time_to_first_token_seconds_count", 0) == 0) count = kv.second;
    }
    CHECK_NEAR(inf, count, 1e-9);
    CHECK_NEAR(count, 1.0, 1e-9);

    /* The bucket layout is vLLM's: the first and last bounds must match. */
    CHECK(r.body.find("vllm:time_to_first_token_seconds_bucket{model_name=\"test-model\","
                      "engine=\"0\",le=\"0.001\"}") != std::string::npos);
    CHECK(r.body.find("le=\"2560\"") != std::string::npos);
}

/* The decode floor reaches /metrics, in seconds. The scheduler keeps nanoseconds and the page
 * divides for display; the wire uses the base unit Prometheus expects, the same choice
 * radiance:engine_step_seconds_* already makes. */
TEST(metrics_carries_the_decode_floor_in_seconds) {
    Fixture f;
    f.sched.m.step_ns_decode_best   = 14708911;   /* 14.71 ms/token: a measured floor */
    f.sched.m.step_ns_decode_recent = 16428860;   /* 16.43 ms/token: the recent window */

    HttpResponse r = f.get("/metrics");
    CHECK_EQ(r.status, 200);
    PromDoc d = parse_prom(r.body);
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:decode_step_seconds_best"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:decode_step_seconds_recent"], std::string("gauge"));

    auto find_one = [&](const std::string& prefix) -> double {
        for (auto& kv : d.samples)
            if (kv.first.rfind(prefix, 0) == 0) return kv.second;
        return -12345.0;
    };
    CHECK_NEAR(find_one("radiance:decode_step_seconds_best{"), 0.014708911, 1e-12);
    CHECK_NEAR(find_one("radiance:decode_step_seconds_recent{"), 0.016428860, 1e-12);

    /* A floor is a duration, so it must survive the number formatter: 0.0147 s rendered as an
     * integer would read as infinitely fast. Checked on a prefix rather than the whole literal
     * because num() prints 17 significant digits and the last ones are the double's own noise. */
    CHECK(r.body.find("radiance:decode_step_seconds_best{model_name=\"test-model\",engine=\"0\"} 0.01470891")
          != std::string::npos);
}

/* Half a window: a floor exists but the recent window has not completed since. Each series is
 * gated on its own value, so one may appear while the other is absent. */
TEST(metrics_omits_only_the_floor_half_that_is_unmeasured) {
    Fixture f;
    f.sched.m.step_ns_decode_best   = 15000000;
    f.sched.m.step_ns_decode_recent = 0;

    HttpResponse r = f.get("/metrics");
    CHECK(r.body.find("radiance:decode_step_seconds_best{") != std::string::npos);
    CHECK(r.body.find("radiance:decode_step_seconds_recent") == std::string::npos);
}

/* The scheduler's totals are cumulative, and a scrape turns them into counter increments by adding
 * the difference from the counter's current value. Concurrent scrapes must still count each total
 * once: however many run, the counter ends at the total. */
TEST(concurrent_scrapes_count_each_total_once) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.steps = 1000;
    s.prefill_tokens = 5000;
    s.draft_depth_seen = 3;
    s.draft_pos_reached[0] = 10;
    s.draft_pos_accepted[0] = 5;

    std::vector<std::thread> scrapers;
    for (int t = 0; t < 8; ++t)
        scrapers.emplace_back([&m, &s] {
            for (int i = 0; i < 2000; ++i) { m.sample(s); if (i % 100 == 0) m.render(); }
        });
    for (auto& t : scrapers) t.join();

    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    auto value = [&d](const std::string& family) {
        for (const auto& kv : d.samples)
            if (kv.first.rfind(family + "{", 0) == 0) return kv.second;
        return -1.0;
    };
    CHECK_EQ(value("radiance:engine_steps_total"), 1000.0);
    CHECK_EQ(value("radiance:prefill_tokens_total"), 5000.0);
}

/* Why snapshots went and why prefix hits reused nothing are counters labelled by reason, so a
 * hybrid server that prefills long prompts again says why. */
TEST(snapshot_evictions_and_cut_hits_are_counted_by_reason) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.prefix_ckpt_retired = 5;  s.prefix_ckpt_lru = 3;  s.prefix_ckpt_refused = 1;
    s.hit_cut_no_snapshot = 2;  s.hit_cut_window = 0;  s.hit_cut_adopt = 4;
    s.hit_cut_tokens = 76092;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:prefix_cache_ckpt_evictions_by_reason_total"],
             std::string("counter"));
    CHECK_EQ(d.type["radiance:prefix_hit_cuts_total"], std::string("counter"));
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"";
    const std::string E = "radiance:prefix_cache_ckpt_evictions_by_reason_total" + L;
    CHECK_NEAR(value(E + ",reason=\"retention\"}"), 5.0, 1e-9);
    CHECK_NEAR(value(E + ",reason=\"store_full\"}"), 3.0, 1e-9);
    CHECK_NEAR(value("radiance:prefix_cache_ckpt_refused_total" + L + "}"), 1.0, 1e-9);
    const std::string C = "radiance:prefix_hit_cuts_total" + L;
    CHECK_NEAR(value(C + ",reason=\"no_snapshot\"}"), 2.0, 1e-9);
    CHECK_NEAR(value(C + ",reason=\"windowed_group\"}"), 0.0, 1e-9);
    CHECK_NEAR(value(C + ",reason=\"adopt_failed\"}"), 4.0, 1e-9);
    CHECK_NEAR(value("radiance:prefix_hit_cut_tokens_total" + L + "}"), 76092.0, 1e-9);
}

/* The decode-only steps --prefill-interleave ran are a counter beside the step counts. */
TEST(the_steps_a_prefill_interleave_held_are_counted) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.steps_held = 6;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:engine_steps_held_total"], std::string("counter"));
    auto it = d.samples.find("radiance:engine_steps_held_total{model_name=\"m\",engine=\"0\"}");
    CHECK(it != d.samples.end() && it->second == 6.0);
}

/* Each draft position's acceptances are a counter next to its reach. At a fixed depth position
 * j+1's reach is j's acceptances, but an adaptive window can end at j with j taken, so a ratio
 * over a window needs both. Here two verifies end at position 1 with it accepted. */
TEST(each_draft_position_counts_its_acceptances) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.draft_depth_seen = 3;
    s.draft_pos_reached[0] = 10; s.draft_pos_accepted[0] = 7;
    s.draft_pos_reached[1] = 7;  s.draft_pos_accepted[1] = 5;
    s.draft_pos_reached[2] = 3;  s.draft_pos_accepted[2] = 2;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:draft_position_accepted_total"], std::string("counter"));
    CHECK_EQ(d.type["radiance:draft_position_reached_total"], std::string("counter"));
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"";
    const std::string A = "radiance:draft_position_accepted_total" + L + ",position=";
    CHECK_NEAR(value(A + "\"0\"}"), 7.0, 1e-9);
    CHECK_NEAR(value(A + "\"1\"}"), 5.0, 1e-9);
    CHECK_NEAR(value(A + "\"2\"}"), 2.0, 1e-9);
    CHECK_NEAR(value(A + "\"3\"}"), -1.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_position_acceptance_ratio" + L + ",position=\"1\"}"),
               5.0 / 7.0, 1e-9);
}

/* The observation ledger reaches /metrics with its two units apart: a cycle hands the predictor
 * one row per draft position, so `rows_*` scale with the window while `cycles_discarded` counts
 * whole cycles that never delivered. Summing them would price the policy by the wrong
 * denominator; the names and the help text say which is which. */
TEST(draft_observation_rows_and_cycles_stay_apart_on_the_wire) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.draft_obs_rows_attempted = 96;
    s.draft_obs_rows_applied = 51;
    s.draft_obs_rows_skipped = 45;
    s.draft_obs_cycles_discarded = 3;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    const char* rows = "radiance:draft_observation_rows_attempted_total";
    const char* cycles = "radiance:draft_observation_cycles_discarded_total";
    CHECK_EQ(d.type[rows], std::string("counter"));
    CHECK_EQ(d.type["radiance:draft_observation_rows_applied_total"], std::string("counter"));
    CHECK_EQ(d.type["radiance:draft_observation_rows_skipped_total"], std::string("counter"));
    CHECK_EQ(d.type[cycles], std::string("counter"));
    CHECK(d.help[rows].find("not one per cycle") != std::string::npos);
    CHECK(d.help[cycles].find("not rows") != std::string::npos);
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"}";
    CHECK_NEAR(value(rows + L), 96.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_observation_rows_applied_total" + L), 51.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_observation_rows_skipped_total" + L), 45.0, 1e-9);
    CHECK_NEAR(value(cycles + L), 3.0, 1e-9);

    /* Cumulative in the scheduler: sample() adds only the difference, so a second scrape of the
     * same totals does not double the counter, and a later total raises it to itself. */
    m.sample(s);
    d = parse_prom(m.render());
    CHECK_NEAR(value(rows + L), 96.0, 1e-9);
    s.draft_obs_rows_attempted = 120;
    m.sample(s);
    d = parse_prom(m.render());
    CHECK_NEAR(value(rows + L), 120.0, 1e-9);
}

/* The adaptive draft window (--draft-window adaptive) is exposed only while it is on, so a
 * deployment without it scrapes exactly what it did before; when on, the gauge and histogram are
 * well-formed families with the labels every other sample carries. */
TEST(the_adaptive_draft_window_is_exposed_only_while_it_is_on) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    m.sample(s);
    std::string body = m.render();
    CHECK(body.find("radiance:draft_depth_adaptive") == std::string::npos);
    CHECK(body.find("radiance:draft_verify_depth_total") == std::string::npos);

    s.draft_adaptive = true;
    s.draft_depth_adaptive = 2.5;
    s.draft_verify_depth[2] = 7;
    s.draft_verify_depth[5] = 3;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:draft_depth_adaptive"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:draft_verify_depth_total"], std::string("counter"));
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"";
    CHECK_NEAR(value("radiance:draft_depth_adaptive" + L + "}"), 2.5, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"1\"}"), 0.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"2\"}"), 7.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"5\"}"), 3.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"6\"}"), -1.0, 1e-9);
}

TEST(models_and_health_answer) {
    Fixture f;
    HttpResponse r = f.get("/v1/models");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("list"));
    CHECK_EQ(j["data"][0]["id"].get<std::string>(), std::string("test-model"));

    HttpResponse one = f.get("/v1/models/test-model");
    CHECK_EQ(one.status, 200);
    /* The entry itself, the way OpenAI's retrieve answers it. */
    CHECK_EQ(json::parse(one.body).value("id", std::string()), std::string("test-model"));
    CHECK_EQ(json::parse(one.body).value("object", std::string()), std::string("model"));
    CHECK_EQ(f.get("/v1/models/other").status, 404);

    HttpResponse h = f.get("/health");
    CHECK_EQ(h.status, 200);
    CHECK(json::parse(h.body)["status"].get<std::string>() == "ok");
}

/* RUNTIME LoRA admin, over a real slot table (lora/slots.h) the way Engine wires it: a request in
 * flight holds its adapter, an unload then drains it (202, new requests 404 as an unknown model,
 * off /v1/models, a load of the name 409), the request's end frees it, and an idle unload is 200.
 * One slot, so the reload also waits out the cooling. */
TEST(an_adapter_in_use_drains_on_unload_and_frees_itself_when_its_last_request_ends) {
    std::atomic<uint64_t> epoch{100};
    lora::SlotTable table;
    table.reset(1, &epoch);
    Fixture f(Fixture::Tweak{ [&](ServerOptions& o, Deps&) {
        o.lora_acquire = [&](const std::string& m, int32_t* s, uint64_t* u, std::shared_ptr<void>* h) {
            return table.acquire(m, s, u, h);
        };
        o.lora_names = [&] { return table.names(); };
        o.lora_load = [&](const std::string& n, const std::string& d, std::string* e) {
            int slot = -1;
            bool cooling = false;
            const int st = table.reserve(n, &slot, &cooling, e);
            if (st < 0) return st;
            table.publish(slot, n, d);
            return (int)RAD_OK;
        };
        o.lora_unload = [&](const std::string& n, std::string* e, int* in_flight) {
            return table.unload(n, in_flight, e);
        };
    } });
    auto listed = [&](const char* id) {
        const json all = json::parse(f.get("/v1/models").body);
        for (const json& m : all["data"])
            if (m.value("id", std::string()) == id) return true;
        return false;
    };
    const std::string load = R"({"lora_name":"ad","lora_path":"/adapters/ad"})";
    const std::string unload = R"({"lora_name":"ad"})";
    const std::string chat_ad = R"({"model":"ad","messages":[{"role":"user","content":"hi"}]})";

    CHECK_EQ(f.post("/v1/load_lora_adapter", load).status, 200);
    CHECK(listed("ad"));

    /* A request through the adapter, held open: the stream never finishes on its own. */
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;
    std::atomic<bool> gone{false};
    {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = R"({"model":"ad","messages":[{"role":"user","content":"hi"}],"stream":true})";
        q.closed = [&gone] { return gone.load(); };
        HttpResponse r = f.srv->router().dispatch(q);
        CHECK_EQ(r.status, 200);
        for (int i = 0; i < 2; ++i) { std::string c; r.next(c); }
        {
            std::lock_guard<std::mutex> lk(f.sched.mu);
            CHECK_EQ(f.sched.seen.back().lora_slot, 0);
            CHECK(f.sched.seen.back().lora_uid != 0);
        }

        /* Unloaded while in use: 202, draining, the one request counted. */
        HttpResponse u = f.post("/v1/unload_lora_adapter", unload);
        CHECK_EQ(u.status, 202);
        const json uj = json::parse(u.body);
        CHECK_EQ(uj.value("action", std::string()), std::string("draining"));
        CHECK_EQ(uj.value("lora_name", std::string()), std::string("ad"));
        CHECK_EQ(uj.value("in_flight", 0), 1);
        /* Asked again: the same, with the count as it is now. */
        HttpResponse u2 = f.post("/v1/unload_lora_adapter", unload);
        CHECK_EQ(u2.status, 202);
        CHECK_EQ(json::parse(u2.body).value("in_flight", 0), 1);

        /* A new request naming it is told what an unknown model is told, and it is not listed. */
        HttpResponse n = f.post("/v1/chat/completions", chat_ad);
        CHECK_EQ(n.status, 404);
        CHECK_EQ(json::parse(n.body)["error"].value("message", std::string()),
                 std::string("no such model or adapter: ad (served: test-model)"));
        CHECK(!listed("ad"));
        CHECK_EQ(f.get("/v1/models/ad").status, 404);
        /* A load of the same name waits for the drain. */
        HttpResponse l = f.post("/v1/load_lora_adapter", load);
        CHECK_EQ(l.status, 409);
        CHECK(has(l.body, "being unloaded"));

        /* The request ends (its client goes away), and with it the last hold. */
        gone.store(true);
        std::string c;
        while (r.next(c)) {}
        r.on_close(false);
    }
    f.sched.join_all();
    CHECK_EQ(table.in_flight("ad"), 0);
    CHECK_EQ(f.post("/v1/unload_lora_adapter", unload).status, 404);   /* freed by itself */

    /* The slot cools before it is written again, then the name loads afresh; idle, it unloads at
     * once. */
    CHECK_EQ(f.post("/v1/load_lora_adapter", load).status, 503);
    epoch += lora::kSlotCoolRounds;
    CHECK_EQ(f.post("/v1/load_lora_adapter", load).status, 200);
    CHECK(listed("ad"));
    HttpResponse idle = f.post("/v1/unload_lora_adapter", unload);
    CHECK_EQ(idle.status, 200);
    CHECK_EQ(json::parse(idle.body).value("action", std::string()), std::string("unloaded"));
    CHECK(!listed("ad"));
}

TEST(the_model_entry_carries_the_context_length_the_server_was_started_with) {
    /* Both routes a client sizes its window from: the list, and the entry it fetches by id. */
    Fixture f(true, 262144, "/models/m.rad");
    CHECK_EQ(json::parse(f.get("/v1/models/test-model").body).value("root", std::string()),
             std::string("/models/m.rad"));
    HttpResponse list = f.get("/v1/models");
    CHECK_EQ(list.status, 200);
    CHECK_EQ(json::parse(list.body)["data"][0].value("max_model_len", (int64_t)0), (int64_t)262144);
    HttpResponse one = f.get("/v1/models/test-model");
    CHECK_EQ(one.status, 200);
    CHECK_EQ(json::parse(one.body).value("max_model_len", (int64_t)0), (int64_t)262144);
    Fixture unknown;
    CHECK(!json::parse(unknown.get("/v1/models").body)["data"][0].contains("max_model_len"));
}

TEST(the_model_entry_and_server_info_report_the_same_capabilities) {
    Fixture f;
    const json info = json::parse(f.get("/server_info").body)["capabilities"];
    const json list = json::parse(f.get("/v1/models").body)["data"][0]["capabilities"];
    const json one  = json::parse(f.get("/v1/models/test-model").body)["capabilities"];
    CHECK(info.is_object());
    CHECK(list == info);
    CHECK(one == info);
    /* Read per request, not frozen at startup: a template that takes no tools says so. */
    CHECK(info.value("tools", false));
    f.tmpl.tools_ok = false;
    CHECK(!json::parse(f.get("/v1/models").body)["data"][0]["capabilities"].value("tools", true));
}

TEST(embeddings_without_an_embedder_is_501_not_an_empty_200) {
    Fixture f;
    HttpResponse r = f.post("/v1/embeddings", R"({"input":"hi"})");
    CHECK_EQ(r.status, 501);
    CHECK(r.body.find("embedding") != std::string::npos);
}

/* ================================================================== partial JSON */

TEST(partial_json_knows_complete_from_truncated) {
    CHECK(json_complete("{}"));
    CHECK(json_complete("  {\"a\":[1,2,{\"b\":null}]}  "));
    CHECK(json_complete("\"hi\""));
    CHECK(json_complete("-1.5e3"));
    CHECK(!json_complete("{"));
    CHECK(!json_complete("{\"a\":"));
    CHECK(!json_complete("{\"a\":1"));
    CHECK(!json_complete("{\"a\":\"unterminated"));
    CHECK(!json_complete("{} trailing"));
    CHECK(!json_complete(""));
}

TEST(a_stable_prefix_never_cuts_a_code_point_or_an_escape) {
    /* Two-byte code point split across a boundary. */
    std::string s = "ab\xc3\xa9";
    CHECK_EQ((long long)json_stable_prefix(s), 4LL);
    CHECK_EQ((long long)json_stable_prefix(s.substr(0, 3)), 2LL);   /* lead byte only */

    /* A lone trailing backslash is held back; an escaped backslash is not. */
    CHECK_EQ((long long)json_stable_prefix("a\\"), 1LL);
    CHECK_EQ((long long)json_stable_prefix("a\\\\"), 3LL);

    /* A half-typed \u escape is held back until all four digits are there. */
    CHECK_EQ((long long)json_stable_prefix("x\\u00"), 1LL);
    CHECK_EQ((long long)json_stable_prefix("x\\u0041"), 7LL);
}

/* A string literal is written exactly as nlohmann writes it and refused exactly where nlohmann
 * refuses: every byte value alone and after each lead byte, and random strings drawn from the
 * deciding bytes (controls, quote and backslash, DEL, continuation and lead bytes). */
TEST(a_quoted_string_is_what_nlohmann_writes) {
    auto agree = [](const std::string& s) {
        std::string got = "x";
        const bool ok = json_quote_append(got, s);
        std::string want;
        bool refused = false;
        try {
            want = "x" + json(s).dump();
        } catch (const nlohmann::json::type_error&) {
            refused = true;
        }
        CHECK_EQ(ok, !refused);
        if (ok) CHECK_EQ(got, want);
        else CHECK_EQ(got, std::string("x"));
        return ok == !refused && (!ok || got == want);
    };
    for (int a = 0; a < 256; ++a) {
        agree(std::string(1, (char)a));
        for (int b = 0; b < 256; ++b) agree(std::string{ (char)a, (char)b, 'z' });
    }
    const unsigned char pool[] = { 0x00, 0x01, 0x08, 0x09, 0x0a, 0x0c, 0x0d, 0x1f, 0x20, '"',
                                   '\\', '/', 'a', 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf,
                                   0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xed, 0xee, 0xef, 0xf0, 0xf4,
                                   0xf5, 0xff };
    std::mt19937 rng(7);
    int bad = 0;
    for (int i = 0; i < 200000 && bad < 4; ++i) {
        std::string s(rng() % 12, '\0');
        for (char& c : s) c = (char)pool[rng() % sizeof pool];
        if (!agree(s)) ++bad;
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the live view */

TEST(the_live_view_prints_ordinary_lines_when_it_is_not_a_terminal) {
    LiveSnapshot s;
    s.step = 12;
    s.model = "m";
    s.running = 4;
    s.waiting = 1;
    s.kv_util = 0.5;
    s.decode_tps = 1234.0;
    s.draft_acceptance = 0.7;
    CardStat c;
    c.index = 0;
    c.name = "R9700";
    c.util = 0.99;
    c.vram_used = 20LL << 30;
    c.vram_total = 32LL << 30;
    s.cards.push_back(c);
    s.experts.n_layers = 2;
    s.experts.n_experts = 4;
    s.experts.tier = {0, 0, 1, 3, 0, 2, 3, 3};
    /* The tally is the snapshot's, not the renderer's. Counting in the renderer would count the
     * cells of the plane, a rectangle a model need not fill. The engine counts the units it
     * placed and both renderers read that, so a snapshot with a plane and no counts renders a
     * plane and no counts. */
    s.experts.n_units = 8;
    s.experts.units[0] = 3; s.experts.units[1] = 1;
    s.experts.units[2] = 1; s.experts.units[3] = 2; s.experts.units[4] = 1;
    s.experts.bytes[0] = 3LL << 30; s.experts.total_bytes = 8LL << 30;
    s.experts.promotions = 40; s.experts.demotions = 37; s.experts.readmits = 12;
    s.link.present = true;
    s.link.world = 2;
    s.link.ar_bytes = 5000;
    s.link.h2d_bytes = 900;
    s.link.d2h_bytes = 800;
    /* The router asked for 200 experts and found 150 resident; the 50 missing cost 700 bytes of
     * GEMM streaming across the link. */
    s.experts.routed = 200; s.experts.routed_resident = 150;
    s.link.stream_bytes = 700;

    /* A pipe is not a terminal, so the constructor picks the line renderer and the frame must
     * contain no escape sequence: a log full of cursor-home codes is ungreppable, and the breakage
     * is invisible on the terminal where it was written (spec §16). */
    LiveView v(nullptr, 1000, stderr);
    std::string line = v.render_line(s);
    CHECK(line.find('\033') == std::string::npos);
    CHECK(line.find("step=12") != std::string::npos);
    CHECK(line.find("run=4") != std::string::npos);
    /* Tokens a second whole, never with a prefix. */
    CHECK(line.find("decode_tps=1234") != std::string::npos);
    CHECK(line.find("draft_accept=0.70") != std::string::npos);
    CHECK(line.find("gpu0=") != std::string::npos);
    CHECK(line.find("experts vram=3") != std::string::npos);
    CHECK(line.find("ssd=2") != std::string::npos);
    CHECK(line.find("mapped=1") != std::string::npos);
    /* Residency as a share, and the churn beside the move totals: a promotion count without the
     * re-admissions cannot tell an engine filling from one thrashing. */
    CHECK(line.find("resident=37.50%") != std::string::npos);
    CHECK(line.find("readmit=12") != std::string::npos);
    /* Hit rate is not residency: 37.50% of the units are in VRAM and 75% of the routing found one
     * there, which is the placement working. Both numbers, since either alone misleads. */
    CHECK(line.find("hit=75.00%") != std::string::npos);
    /* The link, cumulative, with all three of the things that share it. */
    CHECK(line.find("link ar=5000") != std::string::npos);
    CHECK(line.find("h2d=900") != std::string::npos);
    CHECK(line.find("d2h=800") != std::string::npos);
    CHECK(line.find("stream=700") != std::string::npos);
    CHECK(line.back() == '\n');

    /* The terminal renderer is the one that may use escapes, and it colours by tier. */
    std::string screen = v.render_screen(s);
    CHECK(screen.find('\033') != std::string::npos);
    CHECK(screen.find("R9700") != std::string::npos);
    CHECK(screen.find("8 units") != std::string::npos);
    CHECK(screen.find("37.5%") != std::string::npos);
    /* The web page's figures: three significant figures, binary sizes with their unit said once
     * for a used/total pair, SI prefixes on a rate. */
    CHECK(screen.find("20.0 / 32.0 GiB") != std::string::npos);
    CHECK(screen.find("99.0%") != std::string::npos);
    CHECK(screen.find("8.00 GiB") != std::string::npos);
    CHECK(screen.find("3.00 GiB") != std::string::npos);
    CHECK(screen.find("decode 1234 tok/s") != std::string::npos);
    CHECK(screen.find("20.0G") == std::string::npos);
}

/* ================================================================== the sink */

TEST(the_sink_is_a_single_producer_ring_that_fails_loudly_on_overflow) {
    Sink s(16);
    CHECK(!s.done());
    for (int i = 0; i < 16; ++i) CHECK(s.push(65 + i));
    /* One more than capacity is an overflow, and overflow fails the request rather than dropping
     * a token mid-answer. */
    CHECK(!s.push(99));
    CHECK(s.overflowed());
    CHECK(s.done());
    CHECK_EQ((int)s.finish_reason(), (int)Finish::Error);
    CHECK_EQ(s.status(), RAD_E_FULL);

    SinkToken buf[32];
    CHECK_EQ((long long)s.read(buf, 32), 16LL);
    CHECK_EQ(buf[0].token, 65);
    CHECK_EQ(buf[15].token, 80);

    /* The first terminal state wins: a cancel racing an EOS must not report twice. */
    Sink t(16);
    t.finish(Finish::Stop);
    t.finish(Finish::Cancelled);
    CHECK_EQ((int)t.finish_reason(), (int)Finish::Stop);
}

/* ================================================================== the admin surface */

TEST(version_load_and_ping_answer_the_shapes_clients_read) {
    Fixture f;

    HttpResponse v = f.get("/version");
    CHECK_EQ(v.status, 200);
    /* vLLM's key; a client checking compatibility reads exactly this one. */
    CHECK(has(v.body, "\"version\""));

    HttpResponse l = f.get("/load");
    CHECK_EQ(l.status, 200);
    CHECK(has(l.body, "\"server_load\":0"));

    /* Both methods, since load balancers differ in which they send. */
    CHECK_EQ(f.get("/ping").status, 200);
    CHECK_EQ(f.post("/ping", "").status, 200);
    CHECK_EQ(f.get("/health").status, 200);
}

TEST(server_info_lists_the_router_rather_than_a_copy_of_it) {
    Fixture f;
    HttpResponse r = f.get("/server_info");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"model\":\"test-model\""));
    CHECK(has(r.body, "\"chat\":true"));
    /* Every registered route appears, so a route added without a test still shows up here. */
    for (const std::string& sig : f.srv->router().routes()) CHECK(has(r.body, sig));
}

TEST(props_answers_what_a_llama_server_client_reads) {
    Fixture f;
    HttpResponse r = f.get("/props");
    CHECK_EQ(r.status, 200);
    const json p = json::parse(r.body);
    CHECK_EQ(p.value("total_slots", (int64_t)0), (int64_t)4);
    /* No media pipeline attached: no image button. */
    CHECK(!p["modalities"].value("vision", true));
    /* The fixture knows no context length, so none is invented. */
    CHECK(!p["default_generation_settings"].contains("n_ctx"));
    /* The same defaults /server_info reports, under llama-server's names. */
    const json defs = json::parse(f.get("/server_info").body)["request_defaults"];
    const json params = p["default_generation_settings"]["params"];
    CHECK_NEAR(params.value("temperature", -1.0), defs.value("temperature", -2.0), 1e-9);
    CHECK_NEAR(params.value("repeat_penalty", -1.0), defs.value("repetition_penalty", -2.0), 1e-9);
    CHECK_EQ(params.value("n_predict", (int64_t)0), (int64_t)32);   /* the fixture's default */
    CHECK(has(p.value("build_info", std::string()), "radiance"));
}

TEST(props_reports_the_context_one_request_may_use) {
    /* The fixture with a known length, built here rather than widening the shared Fixture. */
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    FakeGrammar gram;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    d.grammar = &gram;
    ServerOptions o;
    o.model_id = "test-model";
    o.max_ctx = 262144;
    Server srv(d, o);
    const json p = json::parse(srv.router().call("GET", "/props").body);
    CHECK_EQ(p["default_generation_settings"].value("n_ctx", (int64_t)0), (int64_t)262144);
    sched.join_all();
}

TEST(server_info_drops_an_engine_blob_that_is_not_an_object) {
    /* A malformed blob must not corrupt the response: /server_info missing a section is better
     * than /server_info being unparseable. */
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.model_id = "test-model";
    o.engine_info = "not json at all";
    Server srv(d, o);
    HttpResponse r = srv.router().call("GET", "/server_info");
    CHECK_EQ(r.status, 200);
    CHECK(!has(r.body, "\"engine\""));

    ServerOptions o2 = o;
    o2.engine_info = R"({"arch":"testarch"})";
    Server srv2(d, o2);
    HttpResponse r2 = srv2.router().call("GET", "/server_info");
    CHECK(has(r2.body, "\"arch\":\"testarch\""));
}

/* /diagnostics describes the host, so without a key it answers this machine only. With a key the
 * transport's gate has already checked the caller, and any address may ask. */
TEST(diagnostics_without_a_key_answers_loopback_only) {
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.model_id = "test-model";
    o.diagnostics_json = [] { return std::string(R"({"findings":[]})"); };
    Server srv(d, o);
    auto from = [](Server& s, const char* peer) {
        HttpRequest q;
        q.path = "/diagnostics";
        q.peer = peer;
        return s.router().dispatch(q).status;
    };
    CHECK_EQ(from(srv, "127.0.0.1:50122"), 200);
    CHECK_EQ(from(srv, "127.8.9.10:50122"), 200);
    CHECK_EQ(from(srv, "::1:50122"), 200);
    CHECK_EQ(from(srv, "[::1]:50122"), 200);
    CHECK_EQ(from(srv, "::ffff:127.0.0.1:50122"), 200);
    CHECK_EQ(from(srv, ""), 200);                       /* in-process */
    CHECK_EQ(from(srv, "192.168.1.20:50122"), 403);
    CHECK_EQ(from(srv, "::ffff:10.0.0.7:50122"), 403);
    CHECK_EQ(from(srv, "fe80::1:50122"), 403);
    CHECK_EQ(from(srv, "128.0.0.1:50122"), 403);

    ServerOptions ok = o;
    ok.api_key = "k";
    Server keyed(d, ok);
    CHECK_EQ(from(keyed, "192.168.1.20:50122"), 200);
}

TEST(tokenize_takes_a_prompt_or_messages_and_never_both) {
    Fixture f;

    HttpResponse r = f.post("/tokenize", R"({"prompt":"AB"})");
    CHECK_EQ(r.status, 200);
    /* FakeTokenizer maps a byte to its code point. */
    CHECK(has(r.body, "\"count\":2"));
    CHECK(has(r.body, "[65,66]"));
    CHECK(has(r.body, "\"token_strs\":null"));

    HttpResponse s = f.post("/tokenize", R"({"prompt":"AB","return_token_strs":true})");
    CHECK(has(s.body, "\"token_strs\":["));

    /* The chat form renders the template first, and the fake's render is not the raw text, so a
     * count equal to the message body would mean the template was skipped. */
    HttpResponse c = f.post("/tokenize", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(c.status, 200);
    CHECK(!has(c.body, "\"count\":2"));

    CHECK_EQ(f.post("/tokenize", R"({"prompt":"A","messages":[]})").status, 400);
    CHECK_EQ(f.post("/tokenize", R"({"tokens":[1]})").status, 400);
    CHECK_EQ(f.post("/tokenize", "[1,2]").status, 400);
}

TEST(a_raw_prompt_parses_control_tokens_on_every_endpoint_that_takes_one) {
    /* A client rendering its own chat template sends "<|im_start|>" meaning the token, as vLLM and
     * llama.cpp read it; /tokenize must count what /v1/completions will see. */
    Fixture f;
    f.tok.last_parse_special = false;
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"<|im_start|>user","max_tokens":4})").status, 200);
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":["a","b"],"max_tokens":4})").status, 200);
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    f.post("/tokenize", R"({"prompt":"<|im_start|>"})");
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    f.post("/tokenize", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK(f.tok.last_parse_special);
    for (const char* body : {R"({"input":"<|im_start|>"})", R"({"input":["a","b"]})"}) {
        OaiRequest r; ApiError e;
        f.tok.last_parse_special = false;
        CHECK_OK(parse_embeddings_request(body, f.odeps(), f.limits(), &r, &e));
        CHECK(f.tok.last_parse_special);
    }
}

/* An embedding model's request fields: prompt_name puts the model's named prompt in front of text
 * and content-part inputs and is refused on token ids and on a name the model does not state;
 * dimensions must be a positive integer; image_tokens one of the trained budgets; a messages input
 * renders through the model's template and excludes `input`. */

TEST(embeddings_take_the_models_named_prompts_and_its_fields) {
    Fixture f;
    FakeEmbedder emb;
    emb.tpl = &f.tmpl;
    OaiDeps d = f.odeps();
    d.embed = &emb;
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":"capital?","prompt_name":"query"})", d,
                                          f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)1);
        CHECK_EQ(r.prompts[0].text, std::string("task: search result | query: capital?"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":["a","b"],"prompt_name":"query"})", d,
                                          f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)2);
        CHECK_EQ(r.prompts[1].text, std::string("task: search result | query: b"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":"x","prompt_name":"nope"})", d, f.limits(),
                                       &r, &e) < 0);
        CHECK_EQ(e.param, std::string("prompt_name"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":[1,2,3],"prompt_name":"query"})", d,
                                       f.limits(), &r, &e) < 0);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":"x","dimensions":256})", d, f.limits(), &r,
                                          &e));
        CHECK_EQ(r.dimensions, (int64_t)256);
        CHECK(parse_embeddings_request(R"({"input":"x","dimensions":0})", d, f.limits(), &r,
                                       &e) < 0);
        CHECK(parse_embeddings_request(R"({"input":"x","image_tokens":300})", d, f.limits(), &r,
                                       &e) < 0);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(
            R"({"messages":[{"role":"user","content":"hello"}]})", d, f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)1);
        CHECK(!r.prompts[0].tokens.empty());
        CHECK(parse_embeddings_request(
                  R"({"input":"x","messages":[{"role":"user","content":"hello"}]})", d,
                  f.limits(), &r, &e) < 0);
    }
}

TEST(detokenize_answers_under_the_key_vllm_uses) {
    Fixture f;
    HttpResponse r = f.post("/detokenize", R"({"tokens":[65,66,67]})");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"prompt\":\"ABC\""));

    CHECK_EQ(f.post("/detokenize", R"({"tokens":"ABC"})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[1,"x"]})").status, 400);
    CHECK_EQ(f.post("/detokenize", "{}").status, 400);
}

/* An id that is not an int32 is refused, not narrowed: 2^40 narrowed is token 0, and the answer
 * would be the text of a token nobody sent. */
TEST(detokenize_refuses_an_id_that_is_not_an_int32) {
    Fixture f;
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[1099511627776]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[65,2147483648]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[-1]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[18446744073709551615]})").status, 400);
}

TEST(tokenizer_info_reports_the_vocabulary_and_what_the_template_can_do) {
    Fixture f;
    HttpResponse r = f.get("/get_tokenizer_info");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"chat_template\":true"));
    CHECK(has(r.body, "\"supports_tools\":true"));
}

TEST(reset_prefix_cache_is_501_when_there_is_no_cache_to_reset) {
    /* The default IScheduler has no prefix cache, and must say so: a 200 here would tell a
     * benchmark it started cold when it did not. */
    Fixture f;
    HttpResponse r = f.post("/reset_prefix_cache", "");
    CHECK_EQ(r.status, 501);
}

/* ================================================================== the admission queue */

/* The Retry-After a 429 carries is the deployment's. */
TEST(a_429_carries_the_deployments_retry_after) {
    FakeScheduler sched;
    FakeTokenizer tok;
    sched.reply = "abcdefgh";
    sched.hang = true;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.max_seqs = 1;
    o.queue_depth = 1;
    o.retry_after_s = 7;
    o.default_max_tokens = 32;
    auto srv = std::make_unique<Server>(d, o);
    std::atomic<bool> never{false};
    std::vector<HttpResponse> live;
    std::string retry;
    for (int i = 0; i < 2; ++i) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/completions";
        q.body = R"({"prompt":"hi","stream":true})";
        q.closed = [&never] { return never.load(); };
        HttpResponse r = srv->router().dispatch(q);
        if (r.status == 429) {
            for (auto& h : r.headers) if (h.first == "Retry-After") retry = h.second;
        } else {
            live.push_back(std::move(r));
        }
    }
    CHECK_EQ(retry, std::string("7"));
    for (auto& r : live) if (r.on_close) r.on_close(false);
    sched.join_all();
    srv.reset();
}

/* ================================================================== llama-server parity */

/* llama-server's spellings reach the engine as the fields they name, and two spellings of one
 * field that disagree are refused. -1 is llama-server's "no limit": not the deployment's default
 * (the fixture's is 32) and not a refusal. */
TEST(llama_server_spellings_reach_the_engine) {
    Fixture f;
    auto post = [&](const std::string& extra) {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}])" + extra + "}");
        f.sched.join_all();
        return r;
    };
    CHECK_EQ(post(R"(,"repeat_penalty":1.25,"n_predict":5)").status, 200);
    {
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_NEAR(f.sched.last_sp.rep_penalty, 1.25, 1e-6);
        CHECK_EQ(f.sched.last_max_tokens, 5);
    }
    CHECK_EQ(post(R"(,"repeat_penalty":1.2,"repetition_penalty":1.3)").status, 400);
    CHECK_EQ(post(R"(,"max_tokens":4,"n_predict":5)").status, 400);
    CHECK_EQ(post("").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK_EQ(f.sched.last_max_tokens, 32); }
    CHECK_EQ(post(R"(,"max_tokens":-1)").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_max_tokens != 32); }
    CHECK_EQ(post(R"(,"max_tokens":0)").status, 400);
}

/* `samplers`: a stage left out is off, temperature after the cuts sets temp_last, and an order the
 * chain cannot run is refused by name, as is a sampler it does not have. */
TEST(the_sampler_order_is_honoured_or_refused) {
    Fixture f;
    auto post = [&](const std::string& extra) {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"temperature":1.5,)"
                                R"("top_p":0.5,"top_k":40)" + extra + "}");
        f.sched.join_all();
        return r;
    };
    CHECK_EQ(post(R"(,"samplers":"kpt")").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_sp.temp_last); }
    CHECK_EQ(post(R"(,"samplers":["top_k","temperature","top_p"])").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(!f.sched.last_sp.temp_last); }
    /* llama.cpp's default list, typ_p before top_p: typical is off, so its place is free. */
    CHECK_EQ(post(R"(,"samplers":["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"])").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_sp.temp_last); }
    CHECK_EQ(post(R"(,"samplers":["top_p"])").status, 200);
    {
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.top_k, 0);                       /* left out: off */
        CHECK_NEAR(f.sched.last_sp.temp, 1.0, 1e-6);
    }
    HttpResponse bad = post(R"(,"min_p":0.1,"samplers":["top_p","temperature","min_p"])");
    CHECK_EQ(bad.status, 400);
    CHECK_EQ(json::parse(bad.body)["error"]["param"].get<std::string>(), std::string("samplers"));
    CHECK_EQ(post(R"(,"samplers":["infill"])").status, 400);
}

/* OpenRouter's session_id travels to the engine as the conversation's label, and nothing else. */
TEST(the_session_id_labels_the_request) {
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"session_id":"conv-7"})").status,
             200);
    f.sched.join_all();
    std::lock_guard<std::mutex> lk(f.sched.mu);
    CHECK_EQ(f.sched.last_label, std::string("conv-7"));
}

/* Assistant prefill: a conversation ending in an assistant message is continued unless the request
 * asks for a generation prompt; asking for both explicitly is refused. */
TEST(a_trailing_assistant_message_is_continued) {
    Fixture f;
    const std::string msgs =
        R"("messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"The"}])";
    CHECK_EQ(f.post("/v1/chat/completions", "{" + msgs + "}").status, 200);
    CHECK_EQ(f.tmpl.last_opt.continue_final, std::string("auto"));
    CHECK(!f.tmpl.last_opt.add_generation_prompt);
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"continue_final_message":"content"})").status, 200);
    CHECK_EQ(f.tmpl.last_opt.continue_final, std::string("content"));
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"add_generation_prompt":true})").status, 200);
    CHECK(f.tmpl.last_opt.continue_final.empty());
    CHECK(f.tmpl.last_opt.add_generation_prompt);
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"continue_final_message":true,"add_generation_prompt":true})")
                 .status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true})")
                 .status, 400);
    f.sched.join_all();
}

/* The deprecated functions API, llama-server's json_schema and reasoning_format reach the
 * template as the fields they spell. */
TEST(older_and_llama_server_chat_spellings_reach_the_template) {
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],
                        "functions":[{"name":"f","parameters":{"type":"object"}}]})").status, 200);
    CHECK_EQ(f.tmpl.last_n_tools, 1);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"functions":[],"tools":[]})")
                 .status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"reasoning_format":"none"})")
                 .status, 200);
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("none"));
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],
                        "reasoning_format":"deepseek-legacy"})").status, 400);
    /* The schema reaches the template as JSON text, to be built into the grammar there. */
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"json_schema":{"type":"object"}})")
                 .status, 200);
    CHECK_EQ(f.tmpl.last_opt.json_schema, std::string(R"({"type":"object"})"));
    f.sched.join_all();
}

/* The routes llama-server answers, answered here too: without /v1, the token count, the rendered
 * prompt, and llama-server's /tokenize and /detokenize shapes. */
TEST(llama_server_routes_answer) {
    Fixture f;
    CHECK_EQ(f.post("/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})").status,
             200);
    CHECK_EQ(f.post("/completions", R"({"prompt":"ab"})").status, 200);
    CHECK_EQ(f.get("/models").status, 200);
    CHECK_EQ(f.get("/v1/health").status, 200);
    f.sched.join_all();

    HttpResponse at = f.post("/apply-template", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(at.status, 200);
    const std::string prompt = json::parse(at.body).value("prompt", std::string());
    CHECK(has(prompt, "<user>hi"));
    HttpResponse it = f.post("/v1/chat/completions/input_tokens",
                             R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(it.status, 200);
    CHECK_EQ(json::parse(it.body).value("input_tokens", (int64_t)-1), (int64_t)prompt.size());

    HttpResponse t = f.post("/tokenize", R"({"content":["ab",7]})");
    CHECK_EQ(t.status, 200);
    CHECK(json::parse(t.body)["tokens"] == json::array({ 97, 98, 7 }));
    HttpResponse tp = f.post("/tokenize", R"({"content":"a","with_pieces":true})");
    CHECK_EQ(json::parse(tp.body)["tokens"][0].value("piece", std::string()), std::string("a"));
    HttpResponse d = f.post("/detokenize", R"({"tokens":[104,105]})");
    CHECK_EQ(json::parse(d.body).value("content", std::string()), std::string("hi"));
    CHECK_EQ(json::parse(d.body).value("prompt", std::string()), std::string("hi"));

    const json p = json::parse(f.get("/props").body);
    CHECK_EQ(p.value("role", std::string()), std::string("model"));
    CHECK(p["default_generation_settings"]["params"]["samplers"].is_array());
    CHECK_EQ(p["default_generation_settings"]["params"].value("mirostat", -1), 0);
}

/* A prompt the context cannot hold names itself in both phrasings that agents compacting on
 * overflow look for, with llama-server's two numbers beside OpenAI's code. */
TEST(an_overflow_reads_as_one_to_the_clients_that_compact) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    FakeGrammar gram;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    d.grammar = &gram;
    ServerOptions o;
    o.model_id = "test-model";
    o.max_ctx = 8;
    Server srv(d, o);
    HttpResponse r = srv.router().call("POST", "/v1/chat/completions",
                                       R"({"messages":[{"role":"user","content":"a long prompt"}]})");
    CHECK_EQ(r.status, 400);
    const json e = json::parse(r.body)["error"];
    CHECK_EQ(e.value("code", std::string()), std::string("context_length_exceeded"));
    CHECK(has(e.value("message", std::string()), "maximum context length is 8 tokens"));
    CHECK(has(e.value("message", std::string()), "exceeds the available context size"));
    CHECK_EQ(e.value("n_ctx", (int64_t)0), (int64_t)8);
    CHECK(e.value("n_prompt_tokens", (int64_t)0) > 8);
    sched.join_all();
}

/* ================================================================== Anthropic's Messages API */

/* The translation is the chat request the Anthropic one means: the system blocks as one system
 * message with Claude Code's billing stamp held still, a tool result as a tool message ahead of
 * the user's own words, the tools and the choice, thinking, stop sequences and the label. */
TEST(an_anthropic_request_becomes_the_chat_request_it_means) {
    OaiLimits lim;
    std::string chat, label;
    ApiError e;
    CHECK_OK(anthropic_to_chat(R"({
        "model":"x","max_tokens":9,"stop_sequences":["END"],
        "system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=1; cch=a5145;"},
                  {"type":"text","text":"Be terse.","cache_control":{"type":"ephemeral"}}],
        "metadata":{"user_id":"user_1_session_9"},
        "thinking":{"type":"disabled"},
        "tools":[{"name":"get","description":"d","input_schema":{"type":"object"}}],
        "tool_choice":{"type":"any","disable_parallel_tool_use":true},
        "messages":[
          {"role":"user","content":"weather?"},
          {"role":"assistant","content":[{"type":"text","text":"Looking."},
                                         {"type":"tool_use","id":"t1","name":"get","input":{"c":"P"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sun"},
                                    {"type":"text","text":"thanks"}]}]})",
        lim, &chat, &label, &e));
    const json c = json::parse(chat);
    CHECK_EQ(c["messages"][0].value("role", ""), std::string("system"));
    CHECK(has(c["messages"][0].value("content", ""), "cch=fffff;"));
    CHECK(has(c["messages"][0].value("content", ""), "Be terse."));
    CHECK_EQ(c["messages"][2]["tool_calls"][0]["function"].value("arguments", ""),
             std::string(R"({"c":"P"})"));
    CHECK_EQ(c["messages"][3].value("role", ""), std::string("tool"));        /* result first */
    CHECK_EQ(c["messages"][3].value("tool_call_id", ""), std::string("t1"));
    CHECK_EQ(c["messages"][4].value("role", ""), std::string("user"));        /* then the words */
    CHECK_EQ(c.value("tool_choice", ""), std::string("required"));
    CHECK(!c.value("parallel_tool_calls", true));
    CHECK(!c.value("enable_thinking", true));
    CHECK_EQ(c["stop"][0].get<std::string>(), std::string("END"));
    CHECK_EQ(label, std::string("user_1_session_9"));

    /* What would change the answer and cannot be served is refused, by name. */
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":[{"type":"document"}]}]})",
                            lim, &chat, &label, &e) < 0);
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],
                                "tools":[{"type":"web_search_20250305","name":"web_search"}]})",
                            lim, &chat, &label, &e) < 0);
    /* A malformed shape is refused, not dropped or passed on as call arguments. */
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"assistant","content":[
                                {"type":"tool_use","id":"t1","name":"f","input":"x"}]}]})",
                            lim, &chat, &label, &e) < 0);
    CHECK_EQ(e.param, std::string("messages[].content[].input"));
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],"tool_choice":"some"})",
                            lim, &chat, &label, &e) < 0);
    CHECK_EQ(e.param, std::string("tool_choice"));
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],"tool_choice":3})",
                            lim, &chat, &label, &e) < 0);
    CHECK_EQ(e.param, std::string("tool_choice"));

    /* Structured output: output_config.format, or the older top-level output_format, is the
     * chat request's json_schema response_format; both at once is refused. */
    for (const char* f : { R"("output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}})",
                           R"("output_format":{"type":"json_schema","schema":{"type":"object"}})" }) {
        CHECK_OK(anthropic_to_chat(std::string(R"({"messages":[{"role":"user","content":"x"}],)") +
                                       f + "}",
                                   lim, &chat, &label, &e));
        const json rf = json::parse(chat)["response_format"];
        CHECK_EQ(rf.value("type", ""), std::string("json_schema"));
        CHECK(rf["json_schema"]["schema"] == json::parse(R"({"type":"object"})"));
    }
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],
                                "output_format":{"type":"json_schema","schema":{}},
                                "output_config":{"format":{"type":"json_schema","schema":{}}}})",
                            lim, &chat, &label, &e) < 0);
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],
                                "output_config":{"format":{"type":"text"}}})",
                            lim, &chat, &label, &e) < 0);
    CHECK_EQ(e.param, std::string("output_config.format"));
}

/* The answer and the errors in Anthropic's shapes, through the server. */
TEST(the_messages_api_answers_in_anthropics_shapes) {
    Fixture f;
    HttpResponse r = f.post("/v1/messages",
                            R"({"model":"x","max_tokens":16,"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    const json m = json::parse(r.body);
    CHECK_EQ(m.value("type", ""), std::string("message"));
    CHECK(has(m.value("id", ""), "msg_"));
    CHECK_EQ(m["content"][0].value("type", ""), std::string("text"));
    CHECK_EQ(m["content"][0].value("text", ""), std::string("hello"));
    CHECK_EQ(m.value("stop_reason", ""), std::string("end_turn"));
    CHECK(m["usage"].contains("input_tokens") && m["usage"].contains("output_tokens"));
    f.sched.join_all();

    HttpResponse bad = f.post("/v1/messages", R"({"model":"x","max_tokens":16})");
    CHECK_EQ(bad.status, 400);
    const json e = json::parse(bad.body);
    CHECK_EQ(e.value("type", ""), std::string("error"));
    CHECK_EQ(e["error"].value("type", ""), std::string("invalid_request_error"));

    HttpResponse n = f.post("/v1/messages/count_tokens",
                            R"({"model":"x","messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(n.status, 200);
    CHECK(json::parse(n.body).value("input_tokens", (int64_t)0) > 0);

    /* The stream: message_start, a text block opened, filled and stopped, then message_delta and
     * message_stop, and no [DONE]. */
    HttpResponse s = f.post("/v1/messages", R"({"model":"x","max_tokens":16,"stream":true,
                                               "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(s.status, 200);
    const std::string body = drain(s);
    std::vector<std::string> types;
    for (size_t at = 0; (at = body.find("event: ", at)) != std::string::npos;) {
        const size_t nl = body.find('\n', at);
        types.push_back(body.substr(at + 7, nl - at - 7));
        at = nl;
    }
    CHECK(!types.empty() && types.front() == "message_start");
    CHECK(types.size() >= 2 && types.back() == "message_stop");
    CHECK(std::find(types.begin(), types.end(), "content_block_start") != types.end());
    CHECK(std::find(types.begin(), types.end(), "message_delta") != types.end());
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();
}

static const char* const kMsgTool = R"({"name":"get_weather","description":"d",
    "input_schema":{"type":"object","properties":{"city":{"type":"string"}}}})";

/* The tests below post through the router, so the fakes see the chat request a Messages request
 * became. */

/* max_tokens is optional, as on the chat path: the reply may take what the context leaves. */
TEST(messages_take_a_request_without_a_max_tokens) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model",
                                      "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    const auto j = json::parse(r.body);
    CHECK_EQ(j["content"][0]["text"].get<std::string>(), std::string("hi there"));
}

/* `fields` posted to /v1/messages beside one user turn. */
static HttpResponse post_messages(Fixture& f, const std::string& fields) {
    return f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,)" + fields +
                                      R"(,"messages":[{"role":"user","content":"hi"}]})");
}

/* A refusal in Anthropic's envelope, naming `field` first, as Anthropic's own do. */
static bool anthropic_refusal(const HttpResponse& r, const std::string& field) {
    const json e = json::parse(r.body, nullptr, false);
    const std::string msg = e.is_object() && e["error"].is_object()
                                ? e["error"].value("message", "") : std::string();
    const bool ok = r.status == 400 && e.value("type", "") == "error" &&
                    e["error"].value("type", "") == "invalid_request_error" &&
                    msg.rfind(field + ":", 0) == 0;
    if (!ok) fprintf(stderr, "    %d %s\n", r.status, r.body.c_str());
    return ok;
}

/* Anthropic's API refuses a field it does not define, and so does this endpoint, in Anthropic's
 * words; the fields it knows and does not serve are refused with the reason. */
TEST(messages_refuse_a_field_they_do_not_know) {
    Fixture f;
    const HttpResponse r = post_messages(f, R"("logprobs":1)");
    CHECK(anthropic_refusal(r, "logprobs"));
    CHECK(has(r.body, "Extra inputs are not permitted"));
    CHECK(anthropic_refusal(post_messages(f, R"("mcp_servers":[{"type":"url","url":"u","name":"n"}])"),
                            "mcp_servers"));
    CHECK(anthropic_refusal(post_messages(f, R"("container":"c")"), "container"));
    CHECK(f.sched.submitted.empty());
}

/* What cannot change the answer here is taken and not acted on: capacity and placement,
 * prompt-cache control, and the inputs of Anthropic's own classifiers and fallbacks. */
TEST(messages_take_the_fields_that_change_nothing_here) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = post_messages(f, R"("speed":"fast","service_tier":"auto",
        "inference_geo":"us","cache_control":{"type":"ephemeral"},"fallbacks":"default",
        "safeguards":[{"type":"t","classifier_context":"c"}],"context_hint":{"enabled":true},
        "diagnostics":{"previous_message_id":null},"thread":{"type":"create"})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["content"][0]["text"].get<std::string>(), std::string("hi there"));
}

/* A thread continuation sends only what follows a stored message, and nothing is stored here,
 * so serving it would answer a conversation missing its start. */
TEST(messages_refuse_to_continue_a_thread) {
    Fixture f;
    CHECK(anthropic_refusal(post_messages(f, R"("thread":{"type":"continue",
                                                          "previous_message_id":"msg_1"})"),
                            "thread"));
    CHECK(anthropic_refusal(post_messages(f, R"("thread":{"type":"fork"})"), "thread.type"));
    CHECK(f.sched.submitted.empty());
}

/* The conversation is served as sent, so a request asking for context edits is answered with
 * Anthropic's report that none applied, in the body and in the stream's message_delta. */
TEST(messages_report_that_no_context_edit_applied) {
    Fixture f;
    f.sched.reply = "hi there";
    const std::string cm = R"("context_management":{"edits":[{"type":"clear_tool_uses_20250919",
        "trigger":{"type":"input_tokens","value":100000},"keep":{"type":"tool_uses","value":4}}]})";
    const HttpResponse r = post_messages(f, cm);
    CHECK_EQ(r.status, 200);
    const json j = json::parse(r.body);
    CHECK_EQ(j["content"][0]["text"].get<std::string>(), std::string("hi there"));
    CHECK(j["context_management"]["applied_edits"] == json::array());
    f.sched.join_all();

    HttpResponse s = post_messages(f, cm + R"(,"stream":true)");
    int bad = 0;
    bool reported = false;
    for (const AnthEvent& e : anth_events(drain(s), &bad))
        if (e.name == "message_delta")
            reported = json::parse(e.payload)["context_management"]["applied_edits"] == json::array();
    CHECK_EQ(bad, 0);
    CHECK(reported);
    f.sched.join_all();

    /* Without it the answer carries no report. */
    CHECK(!json::parse(post_messages(f, R"("stream":false)").body).contains("context_management"));
    CHECK(anthropic_refusal(post_messages(f, R"("context_management":{"edits":{}})"),
                            "context_management.edits"));
}

/* output_config.effort is the chat request's reasoning_effort, which the template reads as its
 * own level; an effort-only system message changes it from the next user turn on. */
TEST(messages_pass_the_effort_to_the_template) {
    Fixture f;
    f.sched.reply = "hi there";
    CHECK_EQ(post_messages(f, R"("output_config":{"effort":"high"})").status, 200);
    CHECK(has(f.tmpl.last_prompt, "[reasoning_effort=\"high\"]"));
    f.sched.join_all();

    const HttpResponse r = f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,
        "output_config":{"effort":"high"},
        "messages":[{"role":"user","content":"plan"},{"role":"assistant","content":"ok"},
                    {"role":"system","content":[],"output_config":{"effort":"low"}},
                    {"role":"user","content":"go"},
                    {"role":"system","content":[],"output_config":{"effort":"max"}}]})");
    CHECK_EQ(r.status, 200);
    CHECK(has(f.tmpl.last_prompt, "[reasoning_effort=\"low\"]"));
    CHECK(has(f.tmpl.last_prompt, "<user>plan<assistant>ok<user>go<assistant>"));
    f.sched.join_all();

    CHECK(anthropic_refusal(post_messages(f, R"("output_config":{"effort":"huge"})"),
                            "output_config.effort"));
    CHECK(anthropic_refusal(post_messages(f, R"("output_config":{"verbosity":1})"),
                            "output_config.verbosity"));
}

/* Thinking: `between_tools` turns it off as `disabled` does; the display and binding settings
 * of Anthropic's signed blocks are taken; another type is refused. */
TEST(messages_read_every_thinking_type) {
    Fixture f;
    f.sched.reply = "fine";
    CHECK_EQ(post_messages(f, R"("thinking":{"type":"between_tools"})").status, 200);
    CHECK(!f.tmpl.last_opt.enable_thinking);
    f.sched.join_all();
    CHECK_EQ(post_messages(f, R"("thinking":{"type":"adaptive","display":"summarized",
                                             "block_binding":{"prefix_mismatch_behavior":"error"}})")
                 .status, 200);
    CHECK(f.tmpl.last_opt.enable_thinking);
    f.sched.join_all();
    CHECK(anthropic_refusal(post_messages(f, R"("thinking":{"type":"deep"})"), "thinking.type"));
    CHECK(anthropic_refusal(post_messages(f, R"("thinking":true)"), "thinking"));
    CHECK(anthropic_refusal(post_messages(f, R"("metadata":"m")"), "metadata"));
    CHECK(anthropic_refusal(post_messages(f, R"("tools":{})"), "tools"));
}

/* A system message after the head of the conversation joins the user message beside it, since
 * templates read a system message only at the head; a conversation may end on one. */
TEST(messages_lower_a_mid_conversation_system_message_to_user_in_place) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "messages":[{"role":"user","content":"hello"},
                                                  {"role":"system",
                                                   "content":[{"type":"text","text":"be brief",
                                                               "cache_control":{"type":"ephemeral"}}]}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_prompt, std::string("<user>hello\nbe brief<assistant>"));
}

/* A system message at the head of the conversation stays the system message. */
TEST(messages_keep_a_leading_system_message_as_the_system_block) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "messages":[{"role":"system","content":"be brief"},
                                                  {"role":"user","content":"hello"}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_prompt, std::string("<system>be brief<user>hello<assistant>"));
}

/* The `system` field and the system messages at the head of the conversation are one system
 * message. */
TEST(messages_join_the_leading_system_messages_into_the_system_block) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "system":"preamble",
                                      "messages":[{"role":"system","content":"first"},
                                                  {"role":"system",
                                                   "content":[{"type":"text","text":"second"}]},
                                                  {"role":"user","content":"hello"}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_prompt,
             std::string("<system>preamble\n\nfirst\n\nsecond<user>hello<assistant>"));
}

/* A turn-scoped system message renders until a later user message exists, then never again;
 * the transcript keeps every copy, so only the newest reminder is read. */
TEST(messages_clear_a_turn_scoped_system_message) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,
        "messages":[{"role":"user","content":"run it"},
                    {"role":"system","clear_at":"next_user_message","content":"old"},
                    {"role":"assistant","content":"ran"},
                    {"role":"user","content":"again"},
                    {"role":"system","clear_at":"next_user_message","content":"new"},
                    {"role":"system","clear_at":"never","content":"kept"}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_prompt,
             std::string("<user>run it<assistant>ran<user>again\nnew\nkept<assistant>"));
    f.sched.join_all();
    const HttpResponse bad = f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,
        "messages":[{"role":"user","content":"hi"},
                    {"role":"system","clear_at":"later","content":"x"}]})");
    CHECK(anthropic_refusal(bad, "messages[].clear_at"));
    /* A tool change carried in a system message would change the tools; it is refused by its
     * block type. */
    const HttpResponse tool = f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,
        "messages":[{"role":"user","content":"hi"},
                    {"role":"system","content":[{"type":"tool_addition","tool":{}}]}]})");
    CHECK(anthropic_refusal(tool, "messages[].content[].type"));
    CHECK(has(tool.body, "tool_addition"));
}

/* A template that enforces alternation, as Gemma 3's does: after an optional system message,
 * user and assistant turns alternate from a user turn, and anything else is refused. */
static const char* const kAlternatingTemplate = R"({{- bos_token -}}
{%- if messages[0]['role'] == 'system' -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    {{- '<turn>' + message['role'] + '\n' -}}
    {%- if message['content'] is string -%}
        {{- message['content'] -}}
    {%- else -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'text' -%}{{- item['text'] -}}{%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '<end>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<turn>assistant\n' -}}{%- endif -%})";

/* The text of a later system message joins the user message beside it (the one after it, else
 * the one before it), so the conversation renders through a template that enforces alternation;
 * only between two assistant turns does it stand as a user message of its own. */
TEST(a_mid_conversation_system_message_keeps_the_roles_alternating) {
    ChatTemplate t;
    REQUIRE(t.load(kAlternatingTemplate, "", "") == RAD_OK);
    OaiLimits lim;
    auto render = [&](const char* body, std::string* prompt) {
        std::string chat, label;
        ApiError e;
        if (anthropic_to_chat(body, lim, &chat, &label, &e) < 0) return -1000;
        ChatPrompt p;
        const int st = t.apply(json::parse(chat)["messages"], json::array(), ChatOptions(), &p);
        if (st < 0) fprintf(stderr, "    %s\n", p.error.c_str());
        *prompt = p.prompt;
        return st;
    };
    std::string p;
    /* After the user's turn, at the end of the conversation. */
    CHECK_EQ(render(R"({"messages":[{"role":"user","content":"hi"},
                                    {"role":"system","content":"be brief"}]})", &p), RAD_OK);
    /* The chat layer joins this template's text blocks with a newline, and a folded text brings
     * its own, so it reads as a paragraph of its own. */
    CHECK(has(p, "<turn>user\nhi\n\nbe brief<end>\n<turn>assistant\n"));
    /* Ahead of the user's turn that follows it, and a run of two. */
    CHECK_EQ(render(R"({"messages":[{"role":"user","content":"plan"},
                                    {"role":"assistant","content":"ok"},
                                    {"role":"system","content":"one"},
                                    {"role":"system","content":[{"type":"text","text":"two"}]},
                                    {"role":"user","content":[{"type":"text","text":"go"}]}]})",
                    &p), RAD_OK);
    CHECK(has(p, "<turn>assistant\nok<end>\n<turn>user\none\n\ntwo\n\ngo<end>\n"));
    /* Between two assistant turns, a user turn of its own keeps the alternation. */
    CHECK_EQ(render(R"({"messages":[{"role":"user","content":"a"},
                                    {"role":"assistant","content":"b"},
                                    {"role":"system","content":"c"},
                                    {"role":"assistant","content":"d"}]})", &p), RAD_OK);
    CHECK(has(p, "<turn>assistant\nb<end>\n<turn>user\nc<end>\n<turn>assistant\nd"));
}

/* Claude Code's billing stamp is held still in a leading system message too, and an empty
 * `system` field is still a system message. */
TEST(messages_hold_the_billing_stamp_in_a_leading_system_message) {
    Fixture f;
    f.sched.reply = "hi there";
    CHECK_EQ(f.post("/v1/messages", R"({"model":"test-model","max_tokens":32,
        "messages":[{"role":"system","content":"x-anthropic-billing-header: cc_version=1; cch=a5145;"},
                    {"role":"user","content":"hi"}]})").status, 200);
    CHECK(has(f.tmpl.last_prompt, "cch=fffff;"));
    f.sched.join_all();
    CHECK_EQ(post_messages(f, R"("system":"")").status, 200);
    CHECK_EQ(f.tmpl.last_prompt, std::string("<system><user>hi<assistant>"));
}

TEST(messages_render_a_turn_and_report_the_usage_split) {
    Fixture f;
    f.sched.reply = "hi there";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,"system":"be brief",
                                      "messages":[{"role":"user","content":"hello"}]})");
    CHECK_EQ(r.status, 200);
    const auto j = json::parse(r.body);
    CHECK(j["id"].get<std::string>().rfind("msg_", 0) == 0);
    CHECK_EQ(j["type"].get<std::string>(), std::string("message"));
    CHECK_EQ(j["role"].get<std::string>(), std::string("assistant"));
    CHECK_EQ(j["model"].get<std::string>(), std::string("test-model"));
    CHECK_EQ(j["content"].size(), (size_t)1);
    CHECK_EQ(j["content"][0]["type"].get<std::string>(), std::string("text"));
    CHECK_EQ(j["content"][0]["text"].get<std::string>(), std::string("hi there"));
    CHECK_EQ(j["stop_reason"].get<std::string>(), std::string("end_turn"));
    CHECK(j["stop_sequence"].is_null());
    /* Anthropic's split: input_tokens without the prefix cache's share, which is
     * cache_read_input_tokens; no cache entry is written for a request, so creation is 0. */
    const auto& u = j["usage"];
    CHECK_EQ(u["input_tokens"].get<int64_t>(), (int64_t)f.tmpl.last_prompt.size() - 4);
    CHECK_EQ(u["cache_read_input_tokens"].get<int64_t>(), (int64_t)4);
    CHECK_EQ(u["cache_creation_input_tokens"].get<int64_t>(), (int64_t)0);
    CHECK_EQ(u["output_tokens"].get<int64_t>(), (int64_t)8);
    CHECK(!u.contains("output_tokens_details"));
    CHECK(f.tmpl.last_prompt.rfind("<system>be brief", 0) == 0);
}

TEST(messages_return_a_tool_call_as_a_tool_use_block) {
    Fixture f;
    f.sched.reply = "TOOL:get_weather:{\"city\":\"Paris\"}";
    const HttpResponse r = f.post("/v1/messages",
                                  std::string(R"({"model":"test-model","max_tokens":128,
                                      "tools":[)") + kMsgTool + R"(],
                                      "messages":[{"role":"user","content":"weather in Paris?"}]})");
    CHECK_EQ(r.status, 200);
    const auto j = json::parse(r.body);
    CHECK_EQ(j["stop_reason"].get<std::string>(), std::string("tool_use"));
    CHECK_EQ(j["content"].size(), (size_t)1);
    const auto& b = j["content"][0];
    CHECK_EQ(b["type"].get<std::string>(), std::string("tool_use"));
    /* The id is the one the chat path mints for the call. */
    CHECK(b["id"].get<std::string>().rfind("call_", 0) == 0);
    CHECK_EQ(b["name"].get<std::string>(), std::string("get_weather"));
    CHECK_EQ(b["input"].dump(), std::string(R"({"city":"Paris"})"));
    CHECK(f.tmpl.last_n_tools == 1);
}

TEST(messages_turns_a_tool_result_back_into_a_tool_message) {
    Fixture f;
    f.sched.reply = "sunny";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "tools":[)" + std::string(kMsgTool) + R"(],
                                      "messages":[
                                          {"role":"user","content":"weather in Paris?"},
                                          {"role":"assistant","content":[
                                              {"type":"text","text":"Let me check."},
                                              {"type":"tool_use","id":"toolu_1",
                                               "name":"get_weather","input":{"city":"Paris"}}]},
                                          {"role":"user","content":[
                                              {"type":"tool_result","tool_use_id":"toolu_1",
                                               "content":"sunny, 21C"}]}]})");
    CHECK_EQ(r.status, 200);
    const auto j = json::parse(r.body);
    CHECK_EQ(j["content"][0]["text"].get<std::string>(), std::string("sunny"));
    /* The tool_result reached the template as a tool message. */
    CHECK_EQ(f.tmpl.last_tool_msgs, 1);
    CHECK(f.tmpl.last_prompt.find("<tool>") != std::string::npos);
    CHECK(f.tmpl.last_prompt.find("sunny, 21C") != std::string::npos);
}

TEST(messages_drop_a_turn_of_redacted_thinking_only) {
    Fixture f;
    f.sched.reply = "fine";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "messages":[
                                          {"role":"user","content":"hi"},
                                          {"role":"assistant","content":[
                                              {"type":"redacted_thinking","data":"AAAA"}]},
                                          {"role":"user","content":"again"}]})");
    CHECK_EQ(r.status, 200);
    /* A turn of redacted thinking only has nothing to render and is left out. */
    CHECK_EQ(f.tmpl.last_prompt, std::string("<user>hi<user>again<assistant>"));
}

TEST(messages_report_the_stop_sequence_that_ended_them) {
    Fixture f;
    f.sched.reply = "hello world";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "stop_sequences":[" world"],
                                      "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    const auto j = json::parse(r.body);
    CHECK_EQ(j["content"][0]["text"].get<std::string>(), std::string("hello"));
    CHECK_EQ(j["stop_reason"].get<std::string>(), std::string("stop_sequence"));
    CHECK_EQ(j["stop_sequence"].get<std::string>(), std::string(" world"));
}

TEST(messages_carry_the_thinking_budget_to_the_sampler) {
    Fixture f;
    f.tmpl.think_open = "T";
    f.tmpl.think_close = "t";
    f.sched.reply = "fine";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "thinking":{"type":"enabled","budget_tokens":2048},
                                      "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(f.tmpl.last_opt.enable_thinking);
    CHECK_EQ(f.sched.last_sp.thinking_budget, (int32_t)2048);
}

TEST(messages_disable_thinking_when_asked) {
    Fixture f;
    f.sched.reply = "fine";
    const HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,
                                      "thinking":{"type":"disabled"},
                                      "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(!f.tmpl.last_opt.enable_thinking);
}

/* Anthropic's event stream: named events, one message_start and one message_stop, block indices
 * over the whole message in the order the blocks open, and no [DONE]. */
TEST(messages_stream_is_the_anthropic_event_sequence) {
    Fixture f;
    f.sched.reply = "hi there";
    HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,"stream":true,
                                      "messages":[{"role":"user","content":"hello"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());
    CHECK(r.content_type.rfind("text/event-stream", 0) == 0);

    const std::string body = drain(r);
    CHECK(body.find("[DONE]") == std::string::npos);
    int bad = 0;
    const std::vector<AnthEvent> evs = anth_events(body, &bad);
    CHECK_EQ(bad, 0);
    REQUIRE(evs.size() >= 5);

    const auto& e0 = evs.front();
    CHECK_EQ(e0.name, std::string("message_start"));
    {
        const auto j = json::parse(e0.payload);
        CHECK_EQ(j["message"]["type"].get<std::string>(), std::string("message"));
        CHECK(j["message"]["id"].get<std::string>().rfind("msg_", 0) == 0);
        CHECK_EQ(j["message"]["usage"]["output_tokens"].get<int64_t>(), (int64_t)0);
        /* The whole prompt; message_delta carries the split. */
        CHECK_EQ(j["message"]["usage"]["input_tokens"].get<int64_t>(),
                 (int64_t)f.tmpl.last_prompt.size());
        CHECK(!j["message"]["usage"].contains("cache_read_input_tokens"));
    }
    CHECK_EQ(evs[1].name, std::string("content_block_start"));
    CHECK_EQ(json::parse(evs[1].payload)["content_block"]["type"].get<std::string>(),
             std::string("text"));
    CHECK_EQ(json::parse(evs[1].payload)["index"].get<int>(), 0);

    /* Deltas for block 0, then its stop, message_delta and message_stop, and nothing else. */
    std::string text;
    size_t i = 2;
    for (; i + 3 < evs.size(); ++i) {
        CHECK_EQ(evs[i].name, std::string("content_block_delta"));
        const auto j = json::parse(evs[i].payload);
        CHECK_EQ(j["index"].get<int>(), 0);
        CHECK_EQ(j["delta"]["type"].get<std::string>(), std::string("text_delta"));
        text += j["delta"]["text"].get<std::string>();
    }
    CHECK(i > 2);
    CHECK_EQ(text, std::string("hi there"));
    CHECK_EQ(evs[i].name, std::string("content_block_stop"));
    CHECK_EQ(json::parse(evs[i].payload)["index"].get<int>(), 0);
    CHECK_EQ(evs[i + 1].name, std::string("message_delta"));
    {
        const auto j = json::parse(evs[i + 1].payload);
        CHECK_EQ(j["delta"]["stop_reason"].get<std::string>(), std::string("end_turn"));
        CHECK(j["delta"]["stop_sequence"].is_null());
        CHECK_EQ(j["usage"]["output_tokens"].get<int64_t>(), (int64_t)8);
        /* The final usage is the split, as in the buffered body. */
        CHECK_EQ(j["usage"]["input_tokens"].get<int64_t>() +
                 j["usage"]["cache_read_input_tokens"].get<int64_t>(),
                 (int64_t)f.tmpl.last_prompt.size());
    }
    CHECK_EQ(evs[i + 2].name, std::string("message_stop"));
    CHECK(i + 3 == evs.size());
}

TEST(messages_stream_a_tool_call_as_blocks) {
    Fixture f;
    f.sched.reply = "TOOL:get_weather:{\"city\":\"Paris\"}";
    HttpResponse r = f.post("/v1/messages",
                                  std::string(R"({"model":"test-model","max_tokens":128,"stream":true,
                                      "tools":[)") + kMsgTool + R"(],
                                      "messages":[{"role":"user","content":"weather in Paris?"}]})");
    CHECK_EQ(r.status, 200);
    int bad = 0;
    const std::vector<AnthEvent> evs = anth_events(drain(r), &bad);
    CHECK_EQ(bad, 0);
    REQUIRE(evs.size() >= 5);
    CHECK_EQ(evs[1].name, std::string("content_block_start"));
    {
        const auto j = json::parse(evs[1].payload);
        CHECK_EQ(j["index"].get<int>(), 0);
        const auto& blk = j["content_block"];
        CHECK_EQ(blk["type"].get<std::string>(), std::string("tool_use"));
        CHECK(blk["id"].get<std::string>().rfind("call_", 0) == 0);
        CHECK_EQ(blk["name"].get<std::string>(), std::string("get_weather"));
        CHECK_EQ(blk["input"], json::object());
    }
    std::string args;
    size_t i = 2;
    for (; i + 3 < evs.size(); ++i) {
        CHECK_EQ(evs[i].name, std::string("content_block_delta"));
        const auto j = json::parse(evs[i].payload);
        CHECK_EQ(j["index"].get<int>(), 0);
        CHECK_EQ(j["delta"]["type"].get<std::string>(), std::string("input_json_delta"));
        args += j["delta"]["partial_json"].get<std::string>();
    }
    CHECK(i > 2);
    CHECK_EQ(args, std::string(R"({"city":"Paris"})"));
    CHECK_EQ(evs[i].name, std::string("content_block_stop"));
    CHECK_EQ(evs[i + 1].name, std::string("message_delta"));
    CHECK_EQ(json::parse(evs[i + 1].payload)["delta"]["stop_reason"].get<std::string>(),
             std::string("tool_use"));
    CHECK_EQ(evs[i + 2].name, std::string("message_stop"));
}

TEST(messages_stream_a_thinking_reply_as_blocks) {
    Fixture f;
    f.tmpl.think_reply = true;
    f.tmpl.think_open = "T";
    f.tmpl.think_close = "t";
    f.sched.reply = "think think\nanswer";
    HttpResponse r = f.post("/v1/messages",
                                  R"({"model":"test-model","max_tokens":32,"stream":true,
                                      "thinking":{"type":"enabled","budget_tokens":2048},
                                      "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    int bad = 0;
    const std::vector<AnthEvent> evs = anth_events(drain(r), &bad);
    CHECK_EQ(bad, 0);

    /* Thinking, then text: the text block is index 1, as indices run over the whole message. */
    REQUIRE(evs.size() >= 7);
    std::string think, text;
    int signatures = 0;
    for (const auto& e : evs) {
        if (e.name != "content_block_delta") continue;
        const auto j = json::parse(e.payload);
        const std::string t = j["delta"]["type"].get<std::string>();
        if (t == "thinking_delta") {
            CHECK_EQ(j["index"].get<int>(), 0);
            think += j["delta"]["thinking"].get<std::string>();
        } else if (t == "signature_delta") {
            /* A thinking block closes with its signature; a local model has none, and the empty
             * one is what llama-server sends. */
            CHECK_EQ(j["index"].get<int>(), 0);
            CHECK_EQ(j["delta"]["signature"].get<std::string>(), std::string(""));
            ++signatures;
        } else {
            CHECK_EQ(t, std::string("text_delta"));
            CHECK_EQ(j["index"].get<int>(), 1);
            text += j["delta"]["text"].get<std::string>();
        }
    }
    CHECK_EQ(signatures, 1);
    CHECK_EQ(think, std::string("think think"));
    CHECK_EQ(text, std::string("answer"));
    /* The blocks opened as their type, in the order above. */
    const auto starts = [&evs](const char* t) {
        int n = 0;
        for (const auto& e : evs)
            if (e.name == "content_block_start" &&
                json::parse(e.payload)["content_block"]["type"].get<std::string>() == t)
                ++n;
        return n;
    };
    CHECK_EQ(starts("thinking"), 1);
    /* A thinking block opens with an empty signature beside the empty text, as Anthropic's
     * stream does; the signature_delta then sets it. */
    for (const auto& e : evs)
        if (e.name == "content_block_start" &&
            json::parse(e.payload)["content_block"]["type"].get<std::string>() == "thinking")
            CHECK(json::parse(e.payload)["content_block"] ==
                  json::parse(R"({"type":"thinking","thinking":"","signature":""})"));
    CHECK_EQ(starts("text"), 1);
}

/* ================================================================== OpenAI's Responses API */

/* The translation: instructions as the system message, input items as their turns (a reasoning
 * item and a call joined into one assistant turn, the call's output as a tool message), function
 * tools only, max_output_tokens, reasoning.effort, text.format, and every field the Responses API
 * does not define passed through. */
TEST(a_responses_request_becomes_the_chat_request_it_means) {
    OaiLimits lim;
    std::string chat, echo;
    ApiError e;
    CHECK_OK(responses_to_chat(R"({
        "model":"m","instructions":"Be terse.","max_output_tokens":7,
        "reasoning":{"effort":"low"},"top_k":5,
        "text":{"format":{"type":"json_schema","name":"a","schema":{"type":"object"}}},
        "tools":[{"type":"function","name":"get","parameters":{"type":"object"}},
                 {"type":"web_search"}],
        "input":[
          {"role":"user","content":"weather?"},
          {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"think"}]},
          {"type":"function_call","call_id":"c1","name":"get","arguments":"{}"},
          {"type":"function_call_output","call_id":"c1","output":"sun"}]})",
        lim, &chat, &echo, &e));
    const json c = json::parse(chat);
    CHECK_EQ(c["messages"][0].value("role", ""), std::string("system"));
    CHECK_EQ(c["messages"][1].value("role", ""), std::string("user"));
    CHECK_EQ(c["messages"][2].value("reasoning_content", ""), std::string("think"));
    CHECK_EQ(c["messages"][2]["tool_calls"][0].value("id", ""), std::string("c1"));
    CHECK_EQ(c["messages"][3].value("role", ""), std::string("tool"));
    CHECK_EQ(c["tools"].size(), (size_t)1);                       /* web_search left out */
    CHECK_EQ(c.value("max_tokens", 0), 7);
    CHECK_EQ(c.value("reasoning_effort", ""), std::string("low"));
    CHECK_EQ(c.value("top_k", 0), 5);                             /* passed through */
    CHECK_EQ(c["response_format"].value("type", ""), std::string("json_schema"));
    CHECK(!c.contains("input") && !c.contains("text") && !c.contains("instructions"));
    CHECK_EQ(json::parse(echo).value("instructions", ""), std::string("Be terse."));

    CHECK(responses_to_chat(R"({"input":"x","previous_response_id":"resp_1"})", lim, &chat,
                            &echo, &e) < 0);
}

/* A namespace groups function tools (Codex sends every MCP server's tools that way). They reach
 * the model flattened to <namespace>__<name>, the reply's function_call item reports the two apart,
 * streamed and not, and the next turn's call and tool_choice come back joined (#119). */
TEST(a_namespaces_tools_are_offered_flat_and_answered_apart) {
    OaiLimits lim;
    std::string chat, echo;
    ApiError e;
    std::map<std::string, std::pair<std::string, std::string>> ns;
    CHECK_OK(responses_to_chat(R"({
        "tools":[{"type":"function","name":"plain","parameters":{"type":"object"}},
                 {"type":"namespace","name":"crm","description":"d","tools":[
                    {"type":"function","name":"find","parameters":{"type":"object"}},
                    {"type":"custom","name":"raw"}]}],
        "tool_choice":{"type":"function","namespace":"crm","name":"find"},
        "input":[{"role":"user","content":"x"},
                 {"type":"function_call","call_id":"c1","namespace":"crm","name":"find","arguments":"{}"},
                 {"type":"function_call_output","call_id":"c1","output":"y"}]})",
        lim, &chat, &echo, &e, &ns));
    const json c = json::parse(chat);
    CHECK_EQ(c["tools"].size(), (size_t)2);                       /* the custom tool left out */
    CHECK_EQ(c["tools"][1]["function"].value("name", ""), std::string("crm__find"));
    CHECK_EQ(c["messages"][1]["tool_calls"][0]["function"].value("name", ""), std::string("crm__find"));
    CHECK_EQ(c["tool_choice"]["function"].value("name", ""), std::string("crm__find"));
    CHECK_EQ(ns.size(), (size_t)1);
    CHECK_EQ(ns["crm__find"].first, std::string("crm"));
    CHECK_EQ(json::parse(echo)["tools"].size(), (size_t)2);      /* echoed as sent */

    /* A flattened name may not shadow a top-level tool's. */
    CHECK(responses_to_chat(R"({"input":"x","tools":[{"type":"function","name":"crm__find"},
        {"type":"namespace","name":"crm","tools":[{"type":"function","name":"find"}]}]})",
        lim, &chat, &echo, &e, &ns) < 0);

    Fixture f;
    f.sched.reply = R"(TOOL:crm__find:{"q":"P"})";
    const std::string tools = R"("tools":[{"type":"namespace","name":"crm","tools":[)"
                              R"({"type":"function","name":"find","parameters":{"type":"object"}}]}],)";
    HttpResponse r = f.post("/v1/responses", "{" + tools + R"("input":"find P"})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o["output"][0].value("type", ""), std::string("function_call"));
    CHECK_EQ(o["output"][0].value("namespace", ""), std::string("crm"));
    CHECK_EQ(o["output"][0].value("name", ""), std::string("find"));
    f.sched.join_all();
    HttpResponse st = f.post("/v1/responses", "{" + tools + R"("stream":true,"input":"find P"})");
    const std::string body = drain(st);
    CHECK(has(body, R"("namespace":"crm")"));
    CHECK(!has(body, "crm__find"));
    f.sched.join_all();
}

TEST(the_responses_api_answers_with_a_response_object_and_its_events) {
    Fixture f;
    HttpResponse r = f.post("/v1/responses", R"({"model":"m","input":"hi"})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o.value("object", ""), std::string("response"));
    CHECK(has(o.value("id", ""), "resp_"));
    CHECK_EQ(o.value("status", ""), std::string("completed"));
    CHECK_EQ(o["output"][0].value("type", ""), std::string("message"));
    CHECK_EQ(o["output"][0]["content"][0].value("text", ""), std::string("hello"));
    CHECK(o["usage"].contains("total_tokens"));
    f.sched.join_all();

    HttpResponse n = f.post("/v1/responses/input_tokens", R"({"model":"m","input":"hi"})");
    CHECK_EQ(json::parse(n.body).value("object", ""), std::string("response.input_tokens"));

    HttpResponse s = f.post("/v1/responses", R"({"model":"m","input":"hi","stream":true})");
    const std::string body = drain(s);
    std::vector<std::string> types;
    int64_t last_seq = -1;
    bool ordered = true;
    for (size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const size_t nl = body.find('\n', at);
        const json ev = json::parse(body.substr(at + 6, nl - at - 6), nullptr, false);
        at = nl;
        if (!ev.is_object()) continue;
        types.push_back(ev.value("type", ""));
        const int64_t q = ev.value("sequence_number", (int64_t)-1);
        ordered = ordered && q == last_seq + 1;
        last_seq = q;
    }
    CHECK(ordered);
    CHECK(!types.empty() && types.front() == "response.created");
    CHECK(!types.empty() && types.back() == "response.completed");
    CHECK(std::find(types.begin(), types.end(), "response.output_text.delta") != types.end());
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();
}

/* ================================================================== llama-server's native shape */

/* /completion and /completions answer in llama-server's native result: the text, how it ended,
 * the counts, the settings it ran with (with the generated ids when asked), and a mixed prompt of
 * strings and ids is one prompt. */
TEST(the_native_completion_answers_in_llama_servers_shape) {
    Fixture f;
    HttpResponse r = f.post("/completion", R"({"prompt":["a",98],"n_predict":10,"return_tokens":true})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o.value("content", ""), std::string("hello"));
    CHECK(o.value("stop", false));
    CHECK_EQ(o.value("stop_type", ""), std::string("eos"));
    CHECK_EQ(o.value("tokens_evaluated", (int64_t)0), (int64_t)2);      /* 'a' and 98: one prompt */
    CHECK_EQ(o["tokens"].size(), (size_t)5);
    CHECK(o.contains("generation_settings") && o["generation_settings"].contains("samplers"));
    f.sched.join_all();

    /* The stream: chunks with the ids that made them, a final result, no [DONE]. */
    HttpResponse s = f.post("/completions", R"({"prompt":"ab","stream":true})");
    const std::string body = drain(s);
    CHECK(has(body, "\"tokens\":["));
    CHECK(has(body, "\"stop\":true"));
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();

    /* return_tokens is the native endpoint's own; elsewhere it is guarded at its off value. */
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"ab","return_tokens":true})").status, 400);
}

/* ================================================================== logprobs */

/* A sink carries each token's alternatives beside it, for a request that enabled them. */
TEST(the_sink_carries_a_tokens_alternatives) {
    Sink s(16);
    s.enable_logprobs(3);
    TokenLogprobs lp;
    lp.logprob = -0.5f;
    lp.n_top = 2;
    lp.top_id[0] = 7; lp.top_lp[0] = -0.1f;
    lp.top_id[1] = 9; lp.top_lp[1] = -2.0f;
    CHECK(s.push(5, &lp));
    CHECK(s.push(6));
    SinkToken buf[4];
    SinkAlt alts[4 * 3];
    CHECK_EQ((long long)s.read(buf, 4, alts), 2LL);
    CHECK_EQ(buf[0].token, 5);
    CHECK_EQ(buf[0].logprob, -0.5f);
    CHECK_EQ(buf[0].n_alt, 2);
    CHECK_EQ(alts[0].id, 7);
    CHECK_EQ(alts[1].id, 9);
    CHECK_EQ(alts[1].logprob, -2.0f);
    CHECK_EQ(buf[1].n_alt, 0);
}

/* Chat: `logprobs` and `top_logprobs` reach the sampler as n_probs, and every token comes back in
 * OpenAI's shape with llama-server's id beside it. */
TEST(chat_logprobs_come_back_per_token_with_their_alternatives) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "logprobs":true,"top_logprobs":2})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.sched.last_sp.n_probs, 2);
    const json c = json::parse(r.body)["choices"][0]["logprobs"]["content"];
    CHECK_EQ(c.size(), (size_t)3);
    CHECK_EQ(c[1]["token"].get<std::string>(), std::string("b"));
    CHECK_EQ(c[1]["id"].get<int>(), (int)'b');
    CHECK_EQ(c[1]["logprob"].get<float>(), -0.5f);
    CHECK_EQ(c[1]["bytes"][0].get<int>(), (int)'b');
    CHECK_EQ(c[1]["top_logprobs"].size(), (size_t)2);
    CHECK_EQ(c[1]["top_logprobs"][1]["token"].get<std::string>(), std::string("y"));
    CHECK_EQ(c[1]["top_logprobs"][1]["logprob"].get<float>(), -2.0f);
    f.sched.join_all();

    /* `logprobs` alone gives the token's own and no alternatives, OpenAI's default of 0. */
    HttpResponse z = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true})");
    CHECK_EQ(f.sched.last_sp.n_probs, 0);
    const json zc = json::parse(z.body)["choices"][0]["logprobs"]["content"];
    CHECK_EQ(zc.size(), (size_t)3);
    CHECK(zc[0]["top_logprobs"].empty());
    f.sched.join_all();

    /* Streamed: each token's entry rides a chunk, and together they are the reply's. */
    HttpResponse s = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                "logprobs":true,"top_logprobs":1})");
    const std::string body = drain(s);
    size_t n = 0;
    for (size_t at = body.find("data: {"); at != std::string::npos;
         at = body.find("data: {", at + 1)) {
        const json ev = json::parse(body.substr(at + 6, body.find("\n\n", at) - at - 6));
        const json& ch = ev["choices"][0];
        if (ch.contains("logprobs") && ch["logprobs"].is_object())
            n += ch["logprobs"]["content"].size();
    }
    CHECK_EQ(n, (size_t)3);
    f.sched.join_all();
}

TEST(logprobs_are_refused_out_of_range_and_with_echo) {
    Fixture f(true, 0, "", true);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,
                        "top_logprobs":21})").status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":2})").status,
             400);
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"hi","logprobs":2,"echo":true})").status,
             400);
    CHECK_EQ(f.post("/completion", R"({"prompt":"hi","n_probs":2,"post_sampling_probs":true})")
                 .status, 400);
}

/* Legacy completions: OpenAI's parallel arrays, with llama-server's content array beside them. */
TEST(completion_logprobs_have_the_legacy_shape) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":2})");
    CHECK_EQ(r.status, 200);
    const json lp = json::parse(r.body)["choices"][0]["logprobs"];
    CHECK_EQ(lp["tokens"].size(), (size_t)3);
    CHECK_EQ(lp["tokens"][2].get<std::string>(), std::string("c"));
    CHECK_EQ(lp["token_logprobs"][2].get<float>(), -0.75f);
    CHECK_EQ(lp["top_logprobs"][0]["x"].get<float>(), -1.0f);
    CHECK_EQ(lp["text_offset"][2].get<int>(), 2);
    CHECK_EQ(lp["content"].size(), (size_t)3);
    f.sched.join_all();
}

/* A stop string's own tokens never went out, so their logprobs do not either. */
TEST(logprobs_end_where_the_stop_string_cut_the_reply) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abXYc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1,"stop":["XY"]})");
    const json o = json::parse(r.body)["choices"][0];
    CHECK_EQ(o["text"].get<std::string>(), std::string("ab"));
    CHECK_EQ(o["logprobs"]["tokens"].size(), (size_t)2);
    f.sched.join_all();
}

/* llama-server's native n_probs: completion_probabilities on the answer, and on each streamed
 * chunk instead when streaming. */
TEST(native_n_probs_return_completion_probabilities) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "ab";
    HttpResponse r = f.post("/completion", R"({"prompt":"hi","n_probs":3})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o["completion_probabilities"].size(), (size_t)2);
    CHECK_EQ(o["completion_probabilities"][0]["top_logprobs"].size(), (size_t)3);
    CHECK_EQ(o["completion_probabilities"][0]["id"].get<int>(), (int)'a');
    CHECK_EQ(o["generation_settings"]["n_probs"].get<int>(), 3);
    f.sched.join_all();

    HttpResponse s = f.post("/completion", R"({"prompt":"hi","n_probs":1,"stream":true})");
    const std::string body = drain(s);
    CHECK(has(body, "\"completion_probabilities\":[{"));
    f.sched.join_all();
}

/* A model with no fill-in-the-middle tokens (the fixture's) cannot infill, and says so. */
TEST(infill_without_fim_tokens_is_refused_by_name) {
    Fixture f;
    HttpResponse r = f.post("/infill", R"({"input_prefix":"def f():\n    ","input_suffix":"\n"})");
    CHECK_EQ(r.status, 501);
    CHECK(has(r.body, "fill-in-the-middle"));
}

/* With the markers, a fill is served; a field of the wrong type, or a body nested past the
 * bound every generation endpoint holds, is a 400. */
TEST(infill_serves_a_fill_and_refuses_a_malformed_body) {
    Fixture f;
    f.tok.fim = true;
    CHECK_EQ(f.post("/infill", R"({"input_prefix":"a","input_suffix":"b",
                                   "input_extra":[{"filename":"x.py","text":"y"}]})").status, 200);
    f.sched.join_all();
    for (const char* body : { R"({"input_prefix":"a","n_predict":"many"})",
                              R"({"input_prefix":"a","max_tokens":1.5})",
                              R"({"input_prefix":"a","input_extra":[{"filename":5}]})",
                              R"({"input_prefix":"a","input_extra":[{"text":["y"]}]})" })
        CHECK_EQ(f.post("/infill", body).status, 400);
    const std::string deep = R"({"input_prefix":"a","input_extra":)" + std::string(2000, '[') +
                             std::string(2000, ']') + "}";
    CHECK_EQ(f.post("/infill", deep).status, 400);
}

/* ================================================================== malformed and failed */

/* A field of the wrong type is a 400 naming it on every API, never a 500: the translators read the
 * client's fields before the chat parser checks them, and the chat parser reads a few before its
 * own validation. */
TEST(a_field_of_the_wrong_type_is_a_400_on_every_api) {
    Fixture f;
    struct Case { const char* path; const char* body; const char* param; };
    for (const Case& c : {
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":1,"content":"hi"}]})",
                   "messages[].role" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":"user",
                   "content":[{"type":"text","text":5}]}]})", "messages[].content[].text" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":"user",
                   "content":[{"type":"image","source":"x"}]}]})", "messages[].content[].source" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"system":[{"type":"text","text":[]}],
                   "messages":[{"role":"user","content":"hi"}]})", "system[].text" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"tool_choice":{"type":"auto",
                   "disable_parallel_tool_use":"yes"},"messages":[{"role":"user","content":"hi"}]})",
                   "disable_parallel_tool_use" },
             Case{ "/v1/chat/completions", R"({"messages":["x",{"role":"assistant","content":"hi"}]})",
                   "messages[]" },
             Case{ "/v1/chat/completions", R"({"messages":[{"role":"user",
                   "content":[{"type":"text","text":5}]}]})", "messages[].content[].text" },
             Case{ "/v1/responses", R"({"input":[{"type":5}]})", "input[].type" },
             Case{ "/v1/responses", R"({"input":[{"role":"user","content":[{"type":"input_text",
                   "text":5}]}]})", "input[].content[].text" },
             Case{ "/v1/responses", R"({"input":[{"type":"function_call","name":"f","arguments":{}}]})",
                   "input[].arguments" },
             Case{ "/v1/responses", R"({"input":"hi","tool_choice":{"type":"function","name":7}})",
                   "tool_choice.name" } }) {
        HttpResponse r = f.post(c.path, c.body);
        CHECK_EQ(r.status, 400);
        const json e = json::parse(r.body, nullptr, false);
        CHECK(e.is_object() && e.contains("error"));
        if (e.is_object() && e["error"].contains("param"))
            CHECK_EQ(e["error"]["param"].get<std::string>(), std::string(c.param));
        else
            CHECK(has(r.body, c.param));
    }
}

/* A stream the engine failed ends in its API's own failure event, never its success event: a
 * client that stops reading at the terminal event must not take the partial reply as whole. */
TEST(a_failed_stream_ends_in_its_apis_own_failure) {
    Fixture f;
    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_UNSUPPORTED;
    f.sched.fail_why = "boom";

    HttpResponse a = f.post("/v1/messages", R"({"model":"x","max_tokens":16,"stream":true,
                                               "messages":[{"role":"user","content":"hi"}]})");
    const std::string ab = drain(a);
    CHECK(has(ab, "event: error\ndata: {\"type\":\"error\""));
    CHECK(has(ab, "boom"));
    CHECK(!has(ab, "message_delta"));
    CHECK(!has(ab, "message_stop"));
    f.sched.join_all();

    HttpResponse r = f.post("/v1/responses", R"({"model":"m","input":"hi","stream":true})");
    const std::string rb = drain(r);
    CHECK(has(rb, "event: response.failed"));
    CHECK(has(rb, "\"status\":\"failed\""));
    CHECK(has(rb, "boom"));
    CHECK(!has(rb, "response.completed"));
    f.sched.join_all();

    HttpResponse c = f.post("/v1/chat/completions", R"({"stream":true,
                                                       "messages":[{"role":"user","content":"hi"}]})");
    const std::string cb = drain(c);
    CHECK(has(cb, "\"finish_reason\":\"error\""));
    CHECK(has(cb, "boom"));
    f.sched.join_all();
}

/* ================================================================== the functions API */

/* OpenAI's deprecated functions API end to end: the reply's call is `function_call` with
 * finish_reason "function_call", streamed and not, one call at most; and the next turn (the
 * assistant's function_call and the `function` message with its result) is served. */
TEST(the_functions_api_answers_and_continues_in_its_own_shape) {
    Fixture f;
    f.sched.reply = R"(TOOL:get:{"c":"P"})";
    const std::string fns = R"("functions":[{"name":"get","parameters":{"type":"object"}}],)";
    HttpResponse r = f.post("/v1/chat/completions",
                            "{" + fns + R"("messages":[{"role":"user","content":"weather?"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(!f.tmpl.last_opt.parallel_tool_calls);
    const json c = json::parse(r.body)["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("function_call"));
    CHECK_EQ(c["message"]["function_call"]["name"].get<std::string>(), std::string("get"));
    CHECK_EQ(c["message"]["function_call"]["arguments"].get<std::string>(),
             std::string(R"({"c":"P"})"));
    CHECK(!c["message"].contains("tool_calls"));
    f.sched.join_all();

    HttpResponse s = f.post("/v1/chat/completions",
                            "{" + fns + R"("stream":true,"messages":[{"role":"user","content":"w"}]})");
    const std::string body = drain(s);
    CHECK(has(body, R"("delta":{"function_call":{"name":"get","arguments":""}})"));
    CHECK(has(body, R"("finish_reason":"function_call")"));
    CHECK(!has(body, "tool_calls"));
    f.sched.join_all();

    f.sched.reply = "sunny";
    HttpResponse t = f.post("/v1/chat/completions", "{" + fns + R"("messages":[
        {"role":"user","content":"weather?"},
        {"role":"assistant","content":null,"function_call":{"name":"get","arguments":"{\"c\":\"P\"}"}},
        {"role":"function","name":"get","content":"sun"}]})");
    CHECK_EQ(t.status, 200);
    CHECK_EQ(json::parse(t.body)["choices"][0]["message"]["content"].get<std::string>(),
             std::string("sunny"));
    f.sched.join_all();
}

/* ================================================================== odds and ends */

/* What the Responses API cannot carry is refused rather than computed and dropped: several
 * choices, and logprobs in any spelling. The chat stream's own extras are dropped. */
TEST(the_responses_api_refuses_what_a_response_cannot_carry) {
    Fixture f(true, 0, "", true);
    for (const char* extra : { R"("n":2)", R"("logprobs":true)", R"("top_logprobs":3)",
                               R"("n_probs":2)", R"("include":["message.output_text.logprobs"])" })
        CHECK_EQ(f.post("/v1/responses", std::string(R"({"input":"hi",)") + extra + "}").status, 400);
    HttpResponse ok = f.post("/v1/responses", R"({"input":"hi","stream":true,"return_progress":true,
                                                  "stream_options":{"include_usage":true},
                                                  "include":["reasoning.encrypted_content"]})");
    CHECK_EQ(ok.status, 200);
    const std::string body = drain(ok);
    CHECK(!has(body, "chat.completion.chunk"));
    CHECK(has(body, "response.completed"));
    f.sched.join_all();
}

/* text_offset is where each token's text starts in the reply, streamed or not, even where a token
 * carries half a character and its own text is empty. */
TEST(text_offset_is_the_same_streamed_and_not_past_a_split_character) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "a\xC3\xA9" "b";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1})");
    CHECK_EQ(r.status, 200);
    const json want = json::array({ 0, 1, 2, 3 });
    CHECK(json::parse(r.body)["choices"][0]["logprobs"]["text_offset"] == want);
    f.sched.join_all();

    HttpResponse s = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1,"stream":true})");
    json got = json::array();
    int bad = 0;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        const json& lp = j["choices"][0]["logprobs"];
        if (lp.is_object()) for (const json& o : lp["text_offset"]) got.push_back(o);
    }
    CHECK_EQ(bad, 0);
    CHECK(got == want);
    f.sched.join_all();
}

/* A piece that is not whole UTF-8 goes as its bytes, 0 to 255, as llama-server sends it. */
TEST(tokenize_pieces_carry_bytes_unsigned) {
    Fixture f;
    HttpResponse r = f.post("/tokenize", R"({"content":"\u00e9","with_pieces":true})");
    CHECK_EQ(r.status, 200);
    const json t = json::parse(r.body)["tokens"];
    CHECK_EQ(t.size(), (size_t)2);
    CHECK(t[0]["piece"] == json::array({ 0xC3 }));
    CHECK(t[1]["piece"] == json::array({ 0xA9 }));
}

/* A conversation label is cut to 128 bytes where a character starts, so it stays text. */
TEST(a_long_session_label_is_cut_on_a_character) {
    Fixture f;
    const std::string id = std::string(127, 'a') + "\xC3\xA9";
    CHECK_EQ(f.post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],
                                               "session_id":")" + id + "\"}").status, 200);
    f.sched.join_all();
    std::lock_guard<std::mutex> lk(f.sched.mu);
    CHECK_EQ(f.sched.last_label, std::string(127, 'a'));
}

/* Anthropic's reply names the model the request asked for, as Anthropic's does. */
TEST(the_messages_api_names_the_requested_model) {
    Fixture f;
    HttpResponse r = f.post("/v1/messages", R"({"model":"claude-x","max_tokens":4,
                                               "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body).value("model", ""), std::string("claude-x"));
    f.sched.join_all();
}

/* ================================================================== the request-key policy */

namespace {

const char* path_of(const std::string& ep) {
    return ep == "chat" ? "/v1/chat/completions" : ep == "completion" ? "/v1/completions"
         : ep == "native" ? "/completion" : "/v1/embeddings";
}

/* A request to `ep` carrying `key: value` and `extra`'s fields beside what the endpoint needs.
 * Returns the status and the field a refusal named. */
int status_with(Fixture& f, const std::string& ep, const std::string& key, const json& value,
                const json& extra, std::string* param) {
    json b = ep == "chat"        ? json::parse(R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":4})")
           : ep == "embedding"   ? json::parse(R"({"input":"hi"})")
                                 : json::parse(R"({"prompt":"hi","max_tokens":4})");
    b[key] = value;
    for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
    HttpResponse r = f.post(path_of(ep), b.dump());
    if (r.streaming()) drain(r);
    f.sched.join_all();
    param->clear();
    const json j = json::parse(r.body, nullptr, false);
    if (r.status >= 400 && j.is_object() && j.contains("error") && j["error"]["param"].is_string())
        *param = j["error"]["param"].get<std::string>();
    return r.status;
}

}  /* namespace */

/* The control vector: absent, the deployment's default applies; an explicit `true` on a
 * deployment with no vector is refused by name (serving the stock model while the client believes
 * otherwise would be a silent lie), and on one with a vector it reaches the engine's request. */
TEST(the_control_vector_field_overrides_the_default_or_is_refused) {
    const json body = json::parse(R"({"messages":[{"role":"user","content":"hi"}],
                                       "max_tokens":4,
                                       "control_vector":true})");
    {
        /* No vector loaded: refused, naming the field. */
        Fixture f;
        HttpResponse r = f.post("/v1/chat/completions", body.dump());
        if (r.streaming()) drain(r);
        f.sched.join_all();
        CHECK_EQ(r.status, 400);
        CHECK(r.body.find("control_vector") != std::string::npos);
        /* The route itself is present and answers 404 on both methods: that is what the page tests
         * for before it hides the panel, and what distinguishes "no vector here" from "not signed
         * in". */
        const HttpResponse g0 = f.get("/admin/control-vector");
        CHECK_EQ(g0.status, 404);
        CHECK(g0.body.find("not_found_error") != std::string::npos);
        CHECK_EQ(f.post("/admin/control-vector", R"({"enabled":true})").status, 404);
    }
    {
        /* A vector loaded, default off: the request's own choice wins and reaches the engine. */
        bool def = false;
        Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) {
            d.control_vector.available   = [] { return true; };
            d.control_vector.get_default = [&def] { return def; };
            d.control_vector.set_default = [&def](bool on) { def = on; };
            d.control_vector.json        = [&def] {
                return std::string("{\"available\":true,\"default\":") +
                       (def ? "true" : "false") + "}";
            };
        } });
        HttpResponse r = f.post("/v1/chat/completions", body.dump());
        if (r.streaming()) drain(r);
        f.sched.join_all();
        CHECK_EQ(r.status, 200);
        CHECK(f.sched.last_sp.use_control_vector);

        /* The same request naming nothing runs the default, which the admin route flips. */
        def = true;
        HttpResponse r2 = f.post("/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],
                                     "max_tokens":4})");
        if (r2.streaming()) drain(r2);
        f.sched.join_all();
        CHECK_EQ(r2.status, 200);
        CHECK(f.sched.last_sp.use_control_vector);

        /* Both spellings in one body: the field's own name wins over Strata's alias, whichever
         * order the parser happens to read them in. */
        HttpResponse r3 = f.post("/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],
                                     "max_tokens":4,
                                     "experimental_speed_projection":true,
                                     "control_vector":false})");
        if (r3.streaming()) drain(r3);
        f.sched.join_all();
        CHECK_EQ(r3.status, 200);
        CHECK(!f.sched.last_sp.use_control_vector);

        /* A JSON null is "not sent", not "false": the deployment's default (on, here) applies. */
        HttpResponse r4 = f.post("/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],
                                     "max_tokens":4,
                                     "control_vector":null})");
        if (r4.streaming()) drain(r4);
        f.sched.join_all();
        CHECK_EQ(r4.status, 200);
        CHECK(f.sched.last_sp.use_control_vector);

        const HttpResponse g = f.get("/admin/control-vector");
        CHECK_EQ(g.status, 200);
        CHECK(g.body.find("\"default\":true") != std::string::npos);
        const HttpResponse p = f.post("/admin/control-vector", R"({"enabled":false})");
        CHECK_EQ(p.status, 200);
        CHECK(!def);
        /* Refuses a body that names no boolean rather than guessing. */
        CHECK_EQ(f.post("/admin/control-vector", R"({"enabled":"yes"})").status, 400);
    }
}

/* The policy oai.cpp's lists state, key by key, on every endpoint: a guarded key is served at its
 * neutral value and refused by name at any other (a 200 means every answer-changing field was
 * honoured); an inert key and an unknown one are accepted and ignored; an accepted key with a
 * valid value is served. The rows are written here, independently of the lists, and every key the
 * lists hold must have one, so a key added there without a decision here fails. */
TEST(every_request_key_is_served_refused_or_ignored_as_its_list_says) {
    FakeEmbedder emb;
    emb.width = 4;
    Fixture f(Fixture::Tweak{ [&](ServerOptions& o, Deps& d) {
        d.embed = &emb;
        o.supports_logprobs = true;
        /* This deployment has a control vector, so `control_vector: true` is served rather
         * than refused; the refusal on a deployment without one is its own case below. */
        d.control_vector.available   = [] { return true; };
        d.control_vector.get_default = [] { return false; };
        d.control_vector.set_default = [](bool) {};
        d.control_vector.json        = [] { return std::string("{\"available\":true}"); };
    } });
    emb.tpl = &f.tmpl;
    f.sched.embedding = { 1.0f, 2.0f, 3.0f, 4.0f };
    const KeyPolicy pol = request_key_policy();
    const std::vector<std::string> eps = { "chat", "completion", "native", "embedding" };
    std::string param;

    /* Guarded: the neutral value, and one that would change the answer. */
    struct Guard { json off, on; };
    const std::map<std::string, Guard> guards = {
        { "logit_bias",          { json::object(), json::parse(R"({"5":10})") } },
        { "mirostat",            { 0, 2 } },
        { "top_n_sigma",         { -1, 1.5 } },
        { "dynatemp_range",      { 0, 0.5 } },
        { "min_keep",            { 0, 3 } },
        { "post_sampling_probs", { false, true } },
        { "best_of",             { 1, 2 } },
        { "suffix",              { "", "x" } },
        { "lora",                { json::array(), json::parse(R"([{"id":0,"scale":1}])") } },
        { "modalities",          { json::parse(R"(["text"])"), json::parse(R"(["text","audio"])") } },
        { "audio",               { nullptr, json::parse(R"({"voice":"x"})") } },
        { "n_indent",            { 0, 4 } },
        { "t_max_predict_ms",    { 0, 100 } },
        { "return_tokens",       { false, true } },
        { "grammar_lazy",        { false, true } },
        { "grammar_triggers",    { json::array(), json::parse(R"([{"type":1,"value":"x"}])") } },
        { "preserved_tokens",    { json::array(), json::parse(R"(["<x>"])") } },
    };
    /* Where a guarded key is the endpoint's own field and served. */
    auto served_on = [](const std::string& key, const std::string& ep) {
        return (key == "return_tokens" && ep == "native") ||
               (key == "preserved_tokens" && (ep == "completion" || ep == "native"));
    };
    for (const std::string& key : pol.guarded) {
        auto g = guards.find(key);
        CHECK(g != guards.end());
        if (g == guards.end()) { fprintf(stderr, "    guarded key with no row: %s\n", key.c_str()); continue; }
        for (const std::string& ep : eps) {
            const int off = status_with(f, ep, key, g->second.off, json::object(), &param);
            CHECK_EQ(off, 200);
            if (off != 200) fprintf(stderr, "    %s off on %s -> %d (%s)\n", key.c_str(), ep.c_str(), off, param.c_str());
            const int on = status_with(f, ep, key, g->second.on, json::object(), &param);
            if (served_on(key, ep)) {
                CHECK_EQ(on, 200);
                continue;
            }
            CHECK_EQ(on, 400);
            CHECK_EQ(param, key);
        }
    }

    /* Inert: accepted with any value, everywhere, and so is an unknown key. */
    std::vector<std::string> ignored = pol.inert;
    for (const std::string& p : pol.inert_prefixes) ignored.push_back(p + "n_max");
    ignored.push_back("a_key_no_list_names");
    for (const std::string& key : ignored)
        for (const std::string& ep : eps) {
            const int s = status_with(f, ep, key, "x", json::object(), &param);
            CHECK_EQ(s, 200);
            if (s != 200) fprintf(stderr, "    ignored key %s on %s -> %d\n", key.c_str(), ep.c_str(), s);
        }

    /* Accepted: a valid value (and the fields it needs beside it) on each endpoint whose list
     * holds it. */
    struct Valid { json value, extra; };
    const json none = json::object();
    const std::map<std::string, Valid> valid = {
        { "model", { "m", none } }, { "stream", { true, none } },
        { "stream_options", { json::parse(R"({"include_usage":true})"), json::parse(R"({"stream":true})") } },
        { "n", { 2, none } }, { "seed", { 1, none } }, { "stop", { json::parse(R"(["zz"])"), none } },
        { "max_tokens", { 4, none } }, { "n_predict", { 4, none } }, { "temperature", { 0.5, none } },
        { "top_p", { 0.9, none } }, { "top_k", { 5, none } }, { "min_p", { 0.1, none } },
        { "typical_p", { 0.9, none } }, { "presence_penalty", { 0.1, none } },
        { "frequency_penalty", { 0.1, none } }, { "repetition_penalty", { 1.1, none } },
        { "repeat_penalty", { 1.1, none } }, { "repeat_last_n", { 32, none } },
        { "xtc_probability", { 0.1, none } }, { "xtc_threshold", { 0.2, none } },
        { "dry_multiplier", { 0.5, none } }, { "dry_base", { 1.75, none } },
        { "dry_allowed_length", { 2, none } }, { "dry_penalty_last_n", { 32, none } },
        { "dry_sequence_breakers", { json::parse(R"(["\n"])"), none } },
        { "samplers", { json::parse(R"(["top_k","temperature"])"), none } },
        { "grammar", { "root ::= \"a\"", none } }, { "json_schema", { json::parse(R"({"type":"object"})"), none } },
        { "priority", { 1, none } }, { "ignore_eos", { false, none } },
        { "timings_per_token", { true, json::parse(R"({"stream":true})") } },
        { "return_progress", { true, json::parse(R"({"stream":true})") } },
        { "session_id", { "s", none } }, { "n_probs", { 2, none } },
        { "messages", { json::parse(R"([{"role":"user","content":"yo"}])"), none } },
        { "tools", { json::parse(R"([{"type":"function","function":{"name":"f"}}])"), none } },
        { "tool_choice", { "auto", none } },
        { "response_format", { json::parse(R"({"type":"json_object"})"), none } },
        { "top_logprobs", { 2, json::parse(R"({"logprobs":true})") } },
        { "parallel_tool_calls", { false, none } }, { "add_generation_prompt", { true, none } },
        { "max_completion_tokens", { 4, none } },
        { "chat_template_kwargs", { json::parse(R"({"x":"y"})"), none } },
        { "reasoning_effort", { "low", none } }, { "enable_thinking", { false, none } },
        { "preserve_thinking", { true, none } }, { "thinking_token_budget", { -1, none } },
        { "thinking_budget", { -1, none } }, { "thinking_budget_tokens", { -1, none } },
        { "reasoning_budget_tokens", { -1, none } }, { "reasoning_format", { "none", none } },
        { "continue_final_message", { false, none } },
        { "functions", { json::parse(R"([{"name":"f"}])"), none } },
        { "function_call", { "auto", none } },
        { "prompt", { "yo", none } }, { "echo", { true, none } },
        { "preserved_tokens", { json::parse(R"(["<x>"])"), none } },
        { "input", { "yo", none } }, { "encoding_format", { "base64", none } },
        { "dimensions", { 2, none } }, { "image_tokens", { 70, none } },
        { "prompt_name", { "query", none } },
        /* The control vector is accepted by every text endpoint under its own name and under
         * Strata's spelling; its semantic refusal (a deployment with no vector) happens at
         * submission, not parsing, so a parse test sees 200 here. */
        { "control_vector", { true, none } },
        { "experimental_speed_projection", { true, none } },
    };
    /* logprobs is spelled per endpoint: true on chat, a count on completions. */
    auto value_for = [&](const std::string& key, const std::string& ep) -> Valid {
        if (key == "logprobs") return { ep == "chat" ? json(true) : json(2), none };
        if (key == "messages" && ep == "embedding")
            return { json::parse(R"([{"role":"user","content":"yo"}])"), json::parse(R"({"input":null})") };
        return valid.at(key);
    };
    auto accept_on = [&](const std::vector<std::string>& keys, const std::vector<std::string>& on) {
        for (const std::string& key : keys) {
            const bool known = key == "logprobs" || valid.count(key);
            CHECK(known);
            if (!known) { fprintf(stderr, "    accepted key with no row: %s\n", key.c_str()); continue; }
            for (const std::string& ep : on) {
                const Valid v = value_for(key, ep);
                const int s = status_with(f, ep, key, v.value, v.extra, &param);
                CHECK_EQ(s, 200);
                if (s != 200) fprintf(stderr, "    %s on %s -> %d (%s)\n", key.c_str(), ep.c_str(), s, param.c_str());
            }
        }
    };
    accept_on(pol.common, { "chat", "completion", "native" });
    accept_on(pol.chat, { "chat" });
    accept_on(pol.completion, { "completion", "native" });
    accept_on(pol.embedding, { "embedding" });

    /* n up to the deployment's bound and not past it. */
    CHECK_EQ(status_with(f, "chat", "n", 8, none, &param), 200);
    CHECK_EQ(status_with(f, "chat", "n", 9, none, &param), 400);
    CHECK_EQ(param, std::string("n"));
}

/* ================================================================== seeds */

/* A seed the request names is the seed it gets; `n` choices draw seed, seed + 1, ... so they
 * differ; a request with no seed, or llama-server's -1, draws a fresh one each time (clients send
 * -1 back from /props as the default, and must not get one fixed sample). Anything else is a 400
 * naming `seed`. */
TEST(a_seed_is_the_requests_or_a_fresh_one_and_each_choice_draws_its_own) {
    Fixture f;
    auto seeds = [&](const char* path, const std::string& body) {
        const size_t before = f.sched.seen.size();
        HttpResponse r = f.post(path, body);
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
        std::vector<uint64_t> s;
        std::lock_guard<std::mutex> lk(f.sched.mu);
        for (size_t i = before; i < f.sched.seen.size(); ++i) s.push_back(f.sched.seen[i].sp.seed);
        return s;
    };
    const std::string chat = R"({"messages":[{"role":"user","content":"hi"}])";
    CHECK(seeds("/v1/chat/completions", chat + R"(,"seed":7,"n":3})") ==
          std::vector<uint64_t>({ 7, 8, 9 }));
    CHECK(seeds("/v1/chat/completions", chat + R"(,"seed":7})") == std::vector<uint64_t>({ 7 }));
    CHECK(seeds("/v1/completions", R"({"prompt":"a","seed":7,"n":2})") ==
          std::vector<uint64_t>({ 7, 8 }));
    for (const std::string& extra : { std::string("}"), std::string(R"(,"seed":-1})") }) {
        const std::vector<uint64_t> a = seeds("/v1/chat/completions", chat + extra);
        const std::vector<uint64_t> b = seeds("/v1/chat/completions", chat + extra);
        REQUIRE(a.size() == 1 && b.size() == 1);
        CHECK(a[0] != b[0]);
        CHECK(a[0] != UINT64_MAX && b[0] != UINT64_MAX);
        const std::vector<uint64_t> two = seeds("/completion", R"({"prompt":"a","n":2)" + extra);
        REQUIRE(two.size() == 2);
        CHECK_EQ(two[1], two[0] + 1);
    }
    for (const char* bad : { R"(,"seed":"7"})", R"(,"seed":1.5})", R"(,"seed":[1]})" }) {
        HttpResponse r = f.post("/v1/chat/completions", chat + bad);
        CHECK_EQ(r.status, 400);
        CHECK_EQ(str_at(json::parse(r.body)["error"], "param"), std::string("seed"));
    }
}

/* ================================================================== media parts in a chat */

namespace {

std::string data_url(const std::string& mime, const std::string& bytes) {
    return "data:" + mime + ";base64," + base64_encode(bytes.data(), bytes.size());
}

std::vector<int32_t> byte_ids(const std::string& s) {
    std::vector<int32_t> v;
    for (unsigned char c : s) v.push_back(c);
    return v;
}

}  /* namespace */

/* Each media part becomes its processor's token run exactly where the template rendered its
 * marker, in order, with its placement recorded at that offset, and the bytes the processor is
 * handed are the part's decoded data. Text around a marker is joined to it with nothing, so no
 * token the model never saw appears beside the picture. */
TEST(a_media_part_reaches_the_prompt_where_the_template_put_it) {
    Fixture f;
    FakeMultimodal mm;
    mm.expand = { 9001, 9002, 9003 };
    OaiDeps d = f.odeps();
    d.mm = &mm;
    OaiLimits lim = f.limits();
    lim.allow_image = lim.allow_audio = lim.allow_video = true;
    const std::string one = "\x89PNG first", two = "\x89PNG second";
    const json body = { { "messages", json::array({ json{ { "role", "user" }, { "content", json::array({
        json{ { "type", "text" }, { "text", "A" } },
        json{ { "type", "image_url" }, { "image_url", { { "url", data_url("image/png", one) } } } },
        json{ { "type", "text" }, { "text", "B" } },
        json{ { "type", "input_audio" }, { "input_audio", { { "data", base64_encode(two.data(), two.size()) },
                                                            { "format", "wav" } } } } }) } } }) } };
    OaiRequest r;
    ApiError e;
    REQUIRE(parse_chat_request(body.dump(), d, lim, &r, &e) == RAD_OK);
    REQUIRE(r.prompts.size() == 1 && r.prompts[0].media);
    std::vector<int32_t> want = byte_ids("<user>A");
    const int32_t at0 = (int32_t)want.size();
    want.insert(want.end(), mm.expand.begin(), mm.expand.end());
    want.push_back('B');
    const int32_t at1 = (int32_t)want.size();
    want.insert(want.end(), mm.expand.begin(), mm.expand.end());
    const std::vector<int32_t> tail = byte_ids("<assistant>");
    want.insert(want.end(), tail.begin(), tail.end());
    CHECK(r.prompts[0].tokens == want);
    REQUIRE(r.prompts[0].media->items.size() == 2);
    CHECK_EQ(r.prompts[0].media->items[0].tok, at0);
    CHECK_EQ(r.prompts[0].media->items[1].tok, at1);
    REQUIRE(mm.got.size() == 2);
    CHECK(mm.got[0].first == MediaKind::Image);
    CHECK(mm.got[0].second == std::vector<uint8_t>(one.begin(), one.end()));
    CHECK(mm.got[1].first == MediaKind::Audio);
    CHECK(mm.got[1].second == std::vector<uint8_t>(two.begin(), two.end()));
}

/* A media part that cannot be served is refused by name: a remote URL (the server fetches nothing
 * on a request's behalf), a data URL that is not base64 or does not decode, an audio format the
 * decoder does not take; a deployment without the encoder answers 501; and a template that drops a
 * part is a 400 rather than a prompt that silently lost its picture. */
TEST(a_media_part_that_cannot_be_served_is_refused_by_name) {
    Fixture f;
    FakeMultimodal mm;
    OaiLimits lim = f.limits();
    lim.allow_image = lim.allow_audio = true;
    auto parse = [&](const json& parts, const OaiDeps& d, const OaiLimits& l, ApiError* e) {
        const json body = { { "messages", json::array({ json{ { "role", "user" }, { "content", parts } } }) } };
        OaiRequest r;
        return parse_chat_request(body.dump(), d, l, &r, e);
    };
    auto image = [](const std::string& url) {
        return json::array({ json{ { "type", "image_url" }, { "image_url", { { "url", url } } } } });
    };
    OaiDeps d = f.odeps();
    d.mm = &mm;
    struct Case { json parts; const char* param; const char* says; };
    const Case cases[] = {
        { image("https://example.com/cat.png"), "messages[].content[].image_url.url", "only data: URLs" },
        { image("data:image/png,xyz"), "messages[].content[].image_url.url", "base64" },
        { image("data:image/png;base64,@@@"), "messages[].content[].image_url.url", "invalid base64" },
        { json::array({ json{ { "type", "input_audio" },
                              { "input_audio", { { "data", "AAAA" }, { "format", "wma" } } } } }),
          "messages[].content[].input_audio.format", "wav" },
    };
    for (const Case& c : cases) {
        ApiError e;
        CHECK_EQ(parse(c.parts, d, lim, &e), RAD_E_INVAL);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string(c.param));
        CHECK(has(e.message, c.says));
    }
    const json ok_image = image("data:image/png;base64,AAAA");
    {
        ApiError e;
        OaiLimits off = lim;
        off.allow_image = false;
        CHECK(parse(ok_image, d, off, &e) < 0);
        CHECK_EQ(e.status, 501);
        OaiDeps no_mm = f.odeps();
        CHECK(parse(ok_image, no_mm, lim, &e) < 0);
        CHECK_EQ(e.status, 501);
        CHECK(parse(json::array({ json{ { "type", "video_url" },
                                        { "video_url", { { "url", "data:video/mp4;base64,AAAA" } } } } }),
                    d, lim, &e) < 0);
        CHECK_EQ(e.status, 501);
    }
    {
        /* The template renders only the first message; the picture was in the second. */
        f.tmpl.drop_after_first = true;
        const json body = { { "messages", json::array({ json{ { "role", "user" }, { "content", "A" } },
                                                        json{ { "role", "user" }, { "content", ok_image } } }) } };
        OaiRequest r;
        ApiError e;
        CHECK_EQ(parse_chat_request(body.dump(), d, lim, &r, &e), RAD_E_INVAL);
        CHECK(has(e.message, "dropped a media part"));
        f.tmpl.drop_after_first = false;
    }
}

/* The media marker is the server's own placeholder, so text that spells it is refused in a
 * request carrying media, as the embeddings endpoint refuses it: otherwise it counts as a picture's
 * place, and with a template that drops a part the picture's tokens land where the user typed it.
 * Without media the marker is plain text and reaches the model as such. */
TEST(text_spelling_the_media_marker_is_refused_beside_media) {
    Fixture f;
    FakeMultimodal mm;
    OaiDeps d = f.odeps();
    d.mm = &mm;
    OaiLimits lim = f.limits();
    lim.allow_image = true;
    const json image = { { "type", "image_url" }, { "image_url", { { "url", "data:image/png;base64,AAAA" } } } };
    const std::string typed = std::string("look: ") + RAD_MEDIA_MARKER + " here";
    auto parse = [&](const json& messages, OaiRequest* r, ApiError* e) {
        return parse_chat_request(json{ { "messages", messages } }.dump(), d, lim, r, e);
    };
    {
        OaiRequest r;
        ApiError e;
        CHECK_EQ(parse(json::array({ json{ { "role", "user" }, { "content", json::array({
                           json{ { "type", "text" }, { "text", typed } }, image }) } } }), &r, &e),
                 RAD_E_INVAL);
        CHECK(has(e.param, "messages"));
        CHECK(has(e.message, RAD_MEDIA_MARKER));
    }
    {
        /* A template that drops the picture's message would leave one marker for one picture:
         * the user's. */
        f.tmpl.drop_after_first = true;
        OaiRequest r;
        ApiError e;
        CHECK_EQ(parse(json::array({ json{ { "role", "user" }, { "content", typed } },
                                     json{ { "role", "user" }, { "content", json::array({ image }) } } }),
                       &r, &e),
                 RAD_E_INVAL);
        CHECK(has(e.message, RAD_MEDIA_MARKER));
        f.tmpl.drop_after_first = false;
    }
    {
        OaiRequest r;
        ApiError e;
        CHECK_OK(parse(json::array({ json{ { "role", "user" }, { "content", typed } } }), &r, &e));
        REQUIRE(r.prompts.size() == 1);
        CHECK(r.prompts[0].media == nullptr);
        CHECK(has(r.prompts[0].text, RAD_MEDIA_MARKER));
    }
}

/* ================================================================== embeddings, end to end */

namespace {

std::vector<float> floats(const json& v) {
    std::vector<float> out;
    for (const json& x : v) out.push_back(x.get<float>());
    return out;
}

}  /* namespace */

/* What /v1/embeddings returns, through the scheduler: `dimensions` truncates and then
 * normalises (OpenAI's and sentence-transformers' order; the other order is not unit length), a
 * zero vector stays zeros rather than NaN, inputs come back in order, base64 carries the same
 * floats, and the bounds and failures are named: wider than the model, an input longer than a
 * step, the engine's failure and a vector of the wrong width. */
TEST(embeddings_are_truncated_then_normalised_and_failures_named) {
    FakeEmbedder emb;
    emb.width = 8;
    emb.longest = 16;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.embed = &emb; } });
    f.sched.embedding = { 3, 4, 0, 12, 0, 0, 0, 0 };
    auto vectors = [&](const std::string& body, int status = 200) {
        HttpResponse r = f.post("/v1/embeddings", body);
        f.sched.join_all();
        CHECK_EQ(r.status, status);
        return json::parse(r.body);
    };
    {
        const std::vector<float> v = floats(vectors(R"({"input":"a","dimensions":3})")["data"][0]["embedding"]);
        REQUIRE(v.size() == 3);
        CHECK_NEAR(v[0], 0.6, 1e-6);
        CHECK_NEAR(v[1], 0.8, 1e-6);
        CHECK_NEAR(v[2], 0.0, 1e-6);
        const std::vector<float> w = floats(vectors(R"({"input":"a"})")["data"][0]["embedding"]);
        REQUIRE(w.size() == 8);
        CHECK_NEAR(w[0], 3.0 / 13, 1e-6);
        CHECK_NEAR(w[3], 12.0 / 13, 1e-6);
    }
    {
        /* The same floats under base64, little-endian f32. */
        const json fl = vectors(R"({"input":"a"})")["data"][0]["embedding"];
        const json b64 = vectors(R"({"input":"a","encoding_format":"base64"})")["data"][0]["embedding"];
        std::vector<uint8_t> raw;
        REQUIRE(base64_decode(b64.get<std::string>(), &raw));
        REQUIRE(raw.size() == 8 * sizeof(float));
        std::vector<float> back(8);
        std::memcpy(back.data(), raw.data(), raw.size());
        CHECK(back == floats(fl));
    }
    {
        f.sched.embed_mark_length = true;
        f.sched.embedding = { 0, 4, 0, 0, 0, 0, 0, 0 };
        const json data = vectors(R"({"input":["a","bbb","cc"]})")["data"];
        REQUIRE(data.size() == 3);
        const int lens[] = { 1, 3, 2 };
        for (int i = 0; i < 3; ++i) {
            CHECK_EQ(data[i]["index"].get<int>(), i);
            CHECK_NEAR(data[i]["embedding"][0].get<float>(), lens[i] / std::sqrt(lens[i] * lens[i] + 16.0), 1e-6);
        }
        f.sched.embed_mark_length = false;
    }
    {
        f.sched.embedding = std::vector<float>(8, 0.0f);
        for (const json& x : vectors(R"({"input":"a"})")["data"][0]["embedding"])
            CHECK_EQ(x.get<float>(), 0.0f);
    }
    f.sched.embedding = { 3, 4, 0, 12, 0, 0, 0, 0 };
    {
        const json e = vectors(R"({"input":"a","dimensions":9})", 400)["error"];
        CHECK_EQ(str_at(e, "param"), std::string("dimensions"));
        CHECK(has(e.value("message", ""), "8 wide"));
        const json l = vectors(R"({"input":")" + std::string(17, 'x') + "\"}", 400)["error"];
        CHECK_EQ(l.value("code", ""), std::string("context_length_exceeded"));
        CHECK(has(l.value("message", ""), "16"));
        vectors(R"({"input":")" + std::string(16, 'x') + "\"}", 200);
    }
    {
        f.sched.fail_status = RAD_E_IO;
        f.sched.fail_why = "the encoder failed";
        CHECK(has(vectors(R"({"input":"a"})", 500)["error"].value("message", ""), "the encoder failed"));
        f.sched.fail_status = 0;
        f.sched.embedding = { 1, 2, 3, 4, 5, 6, 7 };
        CHECK(has(vectors(R"({"input":"a"})", 500)["error"].value("message", ""), "no vector"));
    }
}

/* With a tokenizer that adds BOS, an input of text and a picture gets one BOS in front of the
 * whole, and the picture's placement moves with it. */
TEST(an_embedding_inputs_bos_moves_its_media_by_one) {
    Fixture f;
    f.tok.add_bos = true;
    FakeMultimodal mm;
    mm.expand = { 9001, 9002 };
    FakeEmbedder emb;
    OaiDeps d = f.odeps();
    d.mm = &mm;
    d.embed = &emb;
    OaiLimits lim = f.limits();
    lim.allow_image = true;
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_embeddings_request(R"({"input":[{"type":"text","text":"ab"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]})", d, lim, &r, &e));
    REQUIRE(r.prompts.size() == 1 && r.prompts[0].media);
    CHECK(r.prompts[0].tokens == std::vector<int32_t>({ 1, 'a', 'b', 9001, 9002 }));
    REQUIRE(r.prompts[0].media->items.size() == 1);
    CHECK_EQ(r.prompts[0].media->items[0].tok, 3);
}

/* ================================================================== infill's layout */

/* /infill's prompt is llama-server's repository-level layout, token for token:
 *   [REPO]myproject\n [SEP]name\n text ... [SEP]filename\n [PRE]prefix prompt [SUF]suffix [MID]
 * When it does not fit beside the reply's budget, the extra files are trimmed from the front (the
 * newest kept), never the prefix or suffix, which are the edit; n_predict -1 reserves 256. */
TEST(infill_lays_out_the_repository_and_trims_the_oldest_context_first) {
    const int32_t REPO = 1004, SEP = 1005, PRE = 1001, SUF = 1002, MID = 1003;
    auto bytes = [](const std::string& s) { return byte_ids(s); };
    auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    const std::string body = R"({"input_prefix":"P","input_suffix":"S","prompt":"Q",
        "input_extra":[{"filename":"a.py","text":"A"},{"filename":"b.py","text":"B"}],"n_predict":4})";
    std::vector<int32_t> extra = { REPO };
    extra = cat(extra, bytes("myproject\n"));
    extra.push_back(SEP); extra = cat(extra, bytes("a.py\nA"));
    extra.push_back(SEP); extra = cat(extra, bytes("b.py\nB"));
    extra.push_back(SEP); extra = cat(extra, bytes("filename\n"));
    const std::vector<int32_t> fim = { PRE, 'P', 'Q', SUF, 'S', MID };
    {
        Fixture f;
        f.tok.fim = f.tok.repo = true;
        CHECK_EQ(f.post("/infill", body).status, 200);
        f.sched.join_all();
        REQUIRE(f.sched.seen.size() == 1);
        CHECK(f.sched.seen[0].prompt == cat(extra, fim));
        CHECK_EQ(f.sched.seen[0].max_tokens, 4);
    }
    {
        /* 20 tokens of context: 6 for the edit, 4 for the reply, 10 left for the newest extra. */
        Fixture f(true, 20);
        f.tok.fim = f.tok.repo = true;
        CHECK_EQ(f.post("/infill", body).status, 200);
        f.sched.join_all();
        REQUIRE(f.sched.seen.size() == 1);
        CHECK(f.sched.seen[0].prompt ==
              cat(std::vector<int32_t>(extra.end() - 10, extra.end()), fim));
    }
    {
        /* No limit given: 256 reserved, and a BOS when the tokenizer adds one. */
        Fixture f(true, 300);
        f.tok.fim = f.tok.repo = f.tok.add_bos = true;
        const std::string big = R"({"input_prefix":"P","input_suffix":"S","prompt":"Q","n_predict":-1,
            "input_extra":[{"filename":"a.py","text":")" + std::string(100, 'x') + R"("}]})";
        CHECK_EQ(f.post("/infill", big).status, 200);
        f.sched.join_all();
        REQUIRE(f.sched.seen.size() == 1);
        const std::vector<int32_t>& p = f.sched.seen[0].prompt;
        CHECK_EQ(p.size(), (size_t)(1 + (300 - 256 - 7) + 6));
        CHECK_EQ(p.front(), 1);
        CHECK(std::vector<int32_t>(p.end() - 6, p.end()) == fim);
    }
}

/* ================================================================== echo and batch usage */

/* `echo` puts each prompt in front of its completion, the same streamed as buffered, for one
 * prompt and for a batch. A batch with `n` counts each prompt once and its cached tokens once, and
 * every choice's tokens. llama-server's mixed prompt takes BOS once, before its first element. */
TEST(echo_usage_and_mixed_prompts_count_each_prompt_once) {
    Fixture f;
    f.sched.reply = "xyz";
    {
        const json j = json::parse(f.post("/v1/completions", R"({"prompt":["ab","c"],"echo":true})").body);
        CHECK_EQ(j["choices"][0]["text"].get<std::string>(), std::string("abxyz"));
        CHECK_EQ(j["choices"][1]["text"].get<std::string>(), std::string("cxyz"));
        f.sched.join_all();
        HttpResponse s = f.post("/v1/completions", R"({"prompt":["ab","c"],"echo":true,"stream":true})");
        int bad = 0;
        std::map<int, std::string> text;
        for (const std::string& p : sse_payloads(drain(s), &bad)) {
            if (p == "[DONE]") continue;
            const json c = json::parse(p)["choices"];
            if (!c.empty()) text[c[0]["index"].get<int>()] += c[0]["text"].get<std::string>();
        }
        CHECK_EQ(bad, 0);
        CHECK_EQ(text[0], std::string("abxyz"));
        CHECK_EQ(text[1], std::string("cxyz"));
        f.sched.join_all();
    }
    {
        const json u = json::parse(f.post("/v1/completions", R"({"prompt":["ab","c"],"n":2})").body)["usage"];
        CHECK_EQ(u["prompt_tokens"].get<int>(), 3);
        CHECK_EQ(u["completion_tokens"].get<int>(), 4 * 3);
        CHECK_EQ(u["prompt_tokens_details"]["cached_tokens"].get<int>(), 2 * 4);
        f.sched.join_all();
    }
    {
        f.tok.add_bos = true;
        const size_t before = f.sched.seen.size();
        const json o = json::parse(f.post("/completion", R"({"prompt":["a",98]})").body);
        f.sched.join_all();
        CHECK_EQ(o.value("tokens_evaluated", (int64_t)0), (int64_t)3);
        REQUIRE(f.sched.seen.size() == before + 1);
        CHECK(f.sched.seen[before].prompt == std::vector<int32_t>({ 1, 'a', 98 }));
    }
}

/* ================================================================== the sink, wrapped */

/* The ring keeps each token's alternatives in the slot beside it through a thousand wraps of a
 * 16-slot ring, in batches of every size up to 15, so an off-by-one in the slot arithmetic cannot
 * attach one token's alternatives to another's. */
TEST(the_sink_keeps_each_tokens_alternatives_through_every_wrap) {
    Sink s(16);
    s.enable_logprobs(3);
    CHECK_EQ(s.alt_width(), 3);
    SinkToken buf[16];
    SinkAlt alts[16 * 3];
    std::mt19937 rng(11);
    int32_t pushed = 0, read = 0;
    bool same = true;
    while (read < 1000) {
        const int batch = 1 + (int)(rng() % 15);
        for (int k = 0; k < batch && pushed < 1000; ++k, ++pushed) {
            TokenLogprobs lp;
            lp.logprob = -(float)pushed;
            lp.n_top = pushed % 4;
            for (int j = 0; j < lp.n_top; ++j) {
                lp.top_id[j] = pushed * 10 + j;
                lp.top_lp[j] = -(float)j;
            }
            CHECK(s.push(pushed, &lp));
        }
        const size_t n = s.read(buf, 16, alts);
        for (size_t i = 0; i < n; ++i, ++read) {
            same = same && buf[i].token == read && buf[i].logprob == -(float)read &&
                   buf[i].n_alt == std::min(read % 4, 3);
            for (int j = 0; j < buf[i].n_alt; ++j)
                same = same && alts[i * 3 + (size_t)j].id == read * 10 + j &&
                       alts[i * 3 + (size_t)j].logprob == -(float)j;
        }
    }
    CHECK(same);
    CHECK(!s.overflowed());
}

/* One thread pushes, another reads, through a 16-slot ring: every token arrives once, in order,
 * and the producer never overflows a ring the consumer keeps draining. */
TEST(the_sink_hands_every_token_across_threads_in_order) {
    Sink s(16);
    const int32_t N = 20000;
    std::atomic<int32_t> consumed{0};
    std::thread producer([&] {
        for (int32_t i = 0; i < N; ++i) {
            while (i - consumed.load(std::memory_order_acquire) >= 16) std::this_thread::yield();
            s.push(i);
        }
        s.finish(Finish::Stop);
    });
    SinkToken buf[16];
    int32_t got = 0;
    bool ordered = true;
    for (;;) {
        const size_t n = s.read(buf, 16);
        for (size_t i = 0; i < n; ++i, ++got) ordered = ordered && buf[i].token == got;
        consumed.store(got, std::memory_order_release);
        if (n == 0) {
            if (s.done() && s.n_produced() == got) break;
            std::this_thread::yield();
        }
    }
    producer.join();
    CHECK(ordered);
    CHECK_EQ(got, N);
    CHECK(!s.overflowed());
}

/* ================================================================== error bodies and ids */

/* An error body is valid JSON whatever bytes the request's path held: the 404 names the decoded
 * path, and a byte that is no UTF-8 character must not reach the body as it is, or a strict
 * client cannot parse the refusal. A character that is whole stays as it was sent. */
TEST(an_error_body_is_valid_json_whatever_the_path_held) {
    Fixture f;
    for (const char* path : { "/%FF", "/caf%C3", "/a%C3%28b", "/%ED%A0%80", "/%00%1F" }) {
        HttpResponse r = f.get(path);
        CHECK_EQ(r.status, 404);
        CHECK(json::accept(r.body));
        if (!json::accept(r.body)) fprintf(stderr, "    %s -> %s\n", path, r.body.c_str());
    }
    HttpResponse euro = f.get("/price%E2%82%AC");
    REQUIRE(json::accept(euro.body));
    CHECK(has(json::parse(euro.body)["error"].value("message", ""), "/price\xE2\x82\xAC"));
    HttpResponse m = f.post("/v1/models/%FF", "");
    CHECK(json::accept(m.body));
}

/* A served name may hold a slash (--served-model-name org/model, as the Hub spells models), and
 * the entry is fetched by that id whether the client sends the slash or %2F, as OpenAI's SDK
 * retrieves it. Any other id is a 404. */
TEST(a_served_name_with_a_slash_is_retrieved_by_its_id) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.model_id = "org/model"; } });
    CHECK_EQ(json::parse(f.get("/v1/models").body)["data"][0].value("id", ""), std::string("org/model"));
    for (const char* path : { "/v1/models/org/model", "/v1/models/org%2Fmodel" }) {
        HttpResponse r = f.get(path);
        CHECK_EQ(r.status, 200);
        CHECK_EQ(json::parse(r.body, nullptr, false).value("id", ""), std::string("org/model"));
    }
    for (const char* path : { "/v1/models/org", "/v1/models/org/other", "/v1/models/org/model/x" })
        CHECK_EQ(f.get(path).status, 404);
    Fixture g;
    CHECK_EQ(g.get("/v1/models/test-model").status, 200);
}

/* ================================================================== speech: /v1/audio/transcriptions */

/* A model that transcribes, in a format of the test's own: the reply names its language as
 * "lang <Name>" before "<t>", unless the request forced one. */
struct SpeechRig {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeAudio     audio;
    SpeechFormat  fmt;
    std::unique_ptr<Server> srv;
    explicit SpeechRig(bool transcribes = true, int64_t queue_depth = 0) {
        fmt.transcribe_prompt = transcribes ? "<s>{context}|{audio}|{language}" : "";
        fmt.language_prompt = "lang {language}<t>";
        fmt.transcript_open = "<t>";
        fmt.language_lead = "lang ";
        fmt.languages = { { "en", "English" }, { "zh", "Chinese" } };
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.mm = &audio;
        d.speech = &fmt;
        ServerOptions o;
        o.model_id = "asr-model";
        o.max_seqs = 4;
        o.queue_depth = queue_depth;
        srv.reset(new Server(d, o));
    }
    ~SpeechRig() {
        sched.join_all();
        srv.reset();
    }
    HttpResponse post(std::vector<FormPart> form) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/audio/transcriptions";
        q.form = std::move(form);
        return srv->router().dispatch(std::move(q));
    }
};

static FormPart field(const char* name, const std::string& v) { return { name, "", "", v }; }
static FormPart audio_file(const std::string& bytes) { return { "file", "a.wav", "audio/wav", bytes }; }

/* The reply is read as the model's format says: the language it named before the opener, the
 * transcript after it -- and the request decoded greedily. */
TEST(a_transcription_reads_the_language_and_the_transcript_out_of_the_reply) {
    SpeechRig r;
    r.sched.reply = "lang English<t> hello there";
    HttpResponse res = r.post({ audio_file(tone(8000)), field("model", "asr-model") });
    CHECK_EQ(res.status, 200);
    json j = json::parse(res.body);
    CHECK_EQ(j["text"].get<std::string>(), std::string("hello there"));
    CHECK_EQ(j["usage"]["type"].get<std::string>(), std::string("tokens"));
    CHECK_EQ(j["usage"]["output_tokens"].get<int>(), (int)r.sched.reply.size());
    CHECK(r.sched.last_sp.greedy());

    res = r.post({ audio_file(tone(8000)), field("response_format", "verbose_json") });
    CHECK_EQ(res.status, 200);
    j = json::parse(res.body);
    CHECK_EQ(j["task"].get<std::string>(), std::string("transcribe"));
    CHECK_EQ(j["language"].get<std::string>(), std::string("english"));
    CHECK_NEAR(j["duration"].get<double>(), 0.5, 1e-9);
    CHECK_EQ((int)j["segments"].size(), 1);
    CHECK_EQ(j["segments"][0]["text"].get<std::string>(), std::string("hello there"));

    res = r.post({ audio_file(tone(8000)), field("response_format", "text") });
    CHECK_EQ(res.status, 200);
    CHECK_EQ(res.body, std::string("hello there\n"));
}

/* A forced language is the model's own name for it, from a code or a name; the reply is then the
 * transcript alone. */
TEST(a_forced_language_is_named_as_the_model_names_it) {
    SpeechRig r;
    r.sched.reply = " ni hao ";
    HttpResponse res = r.post({ audio_file(tone(8000)), field("language", "zh"),
                                field("response_format", "verbose_json") });
    CHECK_EQ(res.status, 200);
    json j = json::parse(res.body);
    CHECK_EQ(j["text"].get<std::string>(), std::string("ni hao"));
    CHECK_EQ(j["language"].get<std::string>(), std::string("chinese"));
    res = r.post({ audio_file(tone(8000)), field("language", "ENGLISH") });
    CHECK_EQ(res.status, 200);
}

/* What the route cannot serve is refused by name. */
TEST(a_transcription_it_cannot_serve_is_refused_by_name) {
    SpeechRig r;
    CHECK_EQ(r.post({}).status, 400);
    CHECK_EQ(r.post({ field("model", "asr-model") }).status, 400);
    HttpResponse res = r.post({ audio_file(tone(800)), field("language", "xx") });
    CHECK_EQ(res.status, 400);
    CHECK(has(res.body, "en, zh"));
    CHECK_EQ(r.post({ audio_file(tone(800)), field("response_format", "srt") }).status, 400);
    CHECK_EQ(r.post({ audio_file(tone(800)), field("temperature", "0.7") }).status, 400);
    CHECK_EQ(r.post({ audio_file(tone(800)), field("temperature", "0") }).status, 200);
    res = r.post({ audio_file(tone(800)), field("temperature", "warm") });
    CHECK_EQ(res.status, 400);
    CHECK(has(res.body, "\"param\":\"temperature\""));
    CHECK_EQ(r.post({ audio_file(tone(800)), field("timestamp_granularities[]", "word") }).status,
             400);
    CHECK_EQ(r.post({ audio_file(tone(800)), field("stream", "maybe") }).status, 400);

    SpeechRig deaf(/*transcribes=*/false);
    CHECK_EQ(deaf.post({ audio_file(tone(800)) }).status, 501);
}

/* A long recording is cut at its pauses into clips one pass each, and their transcripts are
 * joined. Clips of at most a second here: the cuts fall in the two silences, and the verbose
 * segments say where. */
TEST(a_long_recording_is_cut_at_its_pauses_and_joined) {
    SpeechRig r;
    r.sched.reply = "lang English<t>hi";
    std::string pcm = tone(40000);
    for (size_t i = 13500; i < 14500; ++i) pcm[i] = 0;
    for (size_t i = 26500; i < 27500; ++i) pcm[i] = 0;
    HttpResponse res = r.post({ audio_file(pcm), field("response_format", "verbose_json") });
    CHECK_EQ(res.status, 200);
    json j = json::parse(res.body);
    CHECK_EQ(j["text"].get<std::string>(), std::string("hi hi hi"));
    REQUIRE_EQ((int)j["segments"].size(), 3);
    REQUIRE_EQ(r.audio.clips.size(), (size_t)3);
    int64_t sum = 0;
    for (int64_t n : r.audio.clips) { CHECK(n <= 16000); sum += n; }
    CHECK_EQ(sum, (int64_t)40000);
    const double c1 = j["segments"][0]["end"].get<double>() * 16000.0;
    const double c2 = j["segments"][1]["end"].get<double>() * 16000.0;
    CHECK(c1 >= 13500 && c1 <= 14500);
    CHECK(c2 >= 26500 && c2 <= 27500);
}

/* A long recording's clips are prepared a wave at a time, each wave no larger than the admission
 * queue, and the request keeps its places from the first wave to the last: with a queue of three,
 * no clip is prepared more than a wave ahead of what has been submitted, and every wave runs. */
TEST(a_long_recording_is_prepared_a_wave_at_a_time_within_the_queue) {
    SpeechRig r(/*transcribes=*/true, /*queue_depth=*/3);
    r.audio.max_clip = 1000;
    std::vector<size_t> prepared;
    r.sched.reply_for = [&](const std::vector<int32_t>&) {
        std::lock_guard<std::mutex> lk(r.audio.mu);
        prepared.push_back(r.audio.clips.size());
        return std::string("lang English<t>w");
    };
    HttpResponse res = r.post({ audio_file(tone(10000)) });
    CHECK_EQ(res.status, 200);
    const size_t clips = r.audio.clips.size();
    CHECK(clips >= 10);
    REQUIRE_EQ(prepared.size(), clips);
    for (size_t k = 0; k < prepared.size(); ++k) CHECK(prepared[k] <= (k / 3 + 1) * 3);
    std::string want;
    for (size_t k = 0; k < clips; ++k) want += k ? " w" : "w";
    CHECK_EQ(json::parse(res.body)["text"].get<std::string>(), want);
    CHECK_EQ(r.srv->in_flight(), (int64_t)0);
}

/* The stream carries the transcript and nothing of the language prefix, and its deltas add up to
 * the text its done event reports. */
TEST(a_streamed_transcription_sends_the_transcript_and_ends_with_done) {
    SpeechRig r;
    r.sched.reply = "lang English<t> hello";
    HttpResponse res = r.post({ audio_file(tone(8000)), field("stream", "true") });
    CHECK_EQ(res.status, 200);
    REQUIRE(res.streaming());
    const std::string all = drain(res);
    std::string deltas, done;
    size_t at = 0;
    while ((at = all.find("data: ", at)) != std::string::npos) {
        const size_t e = all.find("\n\n", at);
        json j = json::parse(all.substr(at + 6, e - at - 6));
        if (j["type"] == "transcript.text.delta") deltas += j["delta"].get<std::string>();
        if (j["type"] == "transcript.text.done") done = j["text"].get<std::string>();
        at = e;
    }
    CHECK_EQ(deltas, std::string("hello"));
    CHECK_EQ(done, std::string("hello"));
    CHECK(!has(all, "lang "));
}
 
/* ================================================================== speech: /v1/realtime */

/* The client's end of a socket, in-process: what the client sends is queued for the route to read,
 * and what the route sends is kept, as events, for the test to wait on. */
struct FakeConn : WsConn {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::pair<std::string, bool>> in;
    bool hung_up = false;
    std::vector<std::string> out;
    bool closed = false;
    int  close_code = 0;
    /* While set, send() blocks: a client that has stopped reading. `held` counts the sends
     * waiting on it. */
    bool hold = false;
    int  held = 0;

    bool read(std::string* msg, bool* binary) override {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !in.empty() || hung_up || closed; });
        if (in.empty()) return false;
        *msg = std::move(in.front().first);
        *binary = in.front().second;
        in.pop_front();
        return true;
    }
    bool send(const std::string& t) override {
        std::unique_lock<std::mutex> lk(m);
        ++held;
        cv.notify_all();
        cv.wait(lk, [&] { return !hold || closed; });
        --held;
        out.push_back(t);
        cv.notify_all();
        return !closed;
    }
    void close(int code, const std::string&) override {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        close_code = code;
        cv.notify_all();
    }

    void client(const json& j) {
        std::lock_guard<std::mutex> lk(m);
        in.emplace_back(j.dump(), false);
        cv.notify_all();
    }
    void client_binary(const std::string& bytes) {
        std::lock_guard<std::mutex> lk(m);
        in.emplace_back(bytes, true);
        cv.notify_all();
    }
    void hangup() {
        std::lock_guard<std::mutex> lk(m);
        hung_up = true;
        cv.notify_all();
    }
    void set_hold(bool h) {
        std::lock_guard<std::mutex> lk(m);
        hold = h;
        cv.notify_all();
    }
    std::vector<json> events() {
        std::lock_guard<std::mutex> lk(m);
        std::vector<json> v;
        for (const auto& o : out) v.push_back(json::parse(o));
        return v;
    }
    /* The `nth` event of `type` (from 0), waiting for it; null when it does not come. */
    json wait_for(const std::string& type, size_t nth = 0, int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            size_t seen = 0;
            for (const auto& o : out) {
                json j = json::parse(o);
                if (j.value("type", "") == type && seen++ == nth) return j;
            }
            if (cv.wait_until(lk, until) == std::cv_status::timeout) return json();
        }
    }
};

struct RealtimeRig {
    SpeechRig rig;
    FakeConn conn;
    std::thread route;
    explicit RealtimeRig(int64_t max_clip = 16000 * 30, int64_t queue_depth = 0)
        : rig(/*transcribes=*/true, queue_depth) {
        rig.audio.max_clip = max_clip;
        rig.audio.per_token = 1600;
        rig.sched.reply_for = asr_reply;
        const SocketHandler* h = rig.srv->router().socket_for("/v1/realtime");
        REQUIRE(h != nullptr);
        route = std::thread([this, h] {
            HttpRequest q;
            q.method = "GET";
            q.path = "/v1/realtime";
            (*h)(q, conn);
        });
    }
    ~RealtimeRig() {
        conn.hangup();
        if (route.joinable()) route.join();
    }
    /* `ms` of tone or silence, a tenth of a second an append, paced so the passes keep up. */
    void speak(int rate, int ms, bool tone, int64_t* phase) {
        for (int t = 0; t < ms; t += 100) {
            conn.client(append_event(pcm16(rate, std::min(100, ms - t), tone, phase)));
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
    }
    /* Every delta for `item`, joined. */
    std::string deltas(const std::string& item) {
        std::string d;
        for (const json& e : conn.events())
            if (e.value("type", "") == "conversation.item.input_audio_transcription.delta" &&
                e.value("item_id", "") == item)
                d += e["delta"].get<std::string>();
        return d;
    }
};

/* Words settle while the audio arrives: deltas come before the turn is committed, they are never
 * taken back -- together they are the start of the completed transcript, and the last of them
 * the rest of it -- and each pass after the first is held to what the earlier ones settled. */
TEST(a_realtime_session_settles_words_while_they_are_spoken) {
    RealtimeRig r;
    json created = r.conn.wait_for("session.created");
    REQUIRE(!created.is_null());
    CHECK_EQ(created["session"]["type"].get<std::string>(), std::string("transcription"));
    CHECK(created["session"]["audio"]["input"]["turn_detection"].is_null());
    r.conn.client(session_update(16000));
    json up = r.conn.wait_for("session.updated");
    REQUIRE(!up.is_null());
    CHECK_EQ(up["session"]["audio"]["input"]["format"]["rate"].get<int>(), 16000);

    int64_t phase = 0;
    r.speak(16000, 3000, true, &phase);
    json first = r.conn.wait_for("conversation.item.input_audio_transcription.delta");
    REQUIRE(!first.is_null());
    const std::string item = first["item_id"].get<std::string>();
    CHECK(r.conn.wait_for("input_audio_buffer.committed", 0, 0).is_null());   /* not yet */
    CHECK(!r.conn.wait_for("radiance.input_audio_transcription.tentative").is_null());

    json commit;
    commit["type"] = "input_audio_buffer.commit";
    r.conn.client(commit);
    json committed = r.conn.wait_for("input_audio_buffer.committed");
    REQUIRE(!committed.is_null());
    CHECK_EQ(committed["item_id"].get<std::string>(), item);
    CHECK(committed["previous_item_id"].is_null());
    json done = r.conn.wait_for("conversation.item.input_audio_transcription.completed");
    REQUIRE(!done.is_null());
    CHECK_EQ(done["item_id"].get<std::string>(), item);
    CHECK_EQ(done["transcript"].get<std::string>(), heard_words(6));
    CHECK_EQ(r.deltas(item), heard_words(6));
    CHECK_EQ(done["usage"]["type"].get<std::string>(), std::string("tokens"));
    CHECK(done["usage"]["output_tokens"].get<int>() > 0);

    /* The last pass was told what was settled: its prompt runs on past the placeholders. */
    std::vector<int32_t> last;
    {
        std::lock_guard<std::mutex> lk(r.rig.sched.mu);
        REQUIRE(r.rig.sched.prompts.size() >= 3);
        last = r.rig.sched.prompts.back();
    }
    std::string tail;
    for (size_t i = last.size(); i > 0 && last[i - 1] != '|'; --i)
        tail.insert(tail.begin(), (char)last[i - 1]);
    CHECK(has(tail, "lang English<t> one"));
    CHECK(r.rig.sched.last_sp.greedy());
}

/* server_vad makes a turn of each stretch of speech: it starts with the speech, less its lead-in,
 * and ends at the pause after it, each turn transcribed and completed on its own. */
TEST(server_vad_makes_a_turn_of_each_stretch_of_speech) {
    RealtimeRig r;
    json vad;
    vad["type"] = "server_vad";
    vad["silence_duration_ms"] = 300;
    vad["prefix_padding_ms"] = 200;
    r.conn.client(session_update(16000, vad));
    REQUIRE(!r.conn.wait_for("session.updated").is_null());
    int64_t phase = 0;
    r.speak(16000, 500, false, &phase);
    r.speak(16000, 1500, true, &phase);
    r.speak(16000, 600, false, &phase);
    r.speak(16000, 1000, true, &phase);
    r.speak(16000, 600, false, &phase);

    json s0 = r.conn.wait_for("input_audio_buffer.speech_started", 0);
    json e0 = r.conn.wait_for("input_audio_buffer.speech_stopped", 0);
    json s1 = r.conn.wait_for("input_audio_buffer.speech_started", 1);
    json e1 = r.conn.wait_for("input_audio_buffer.speech_stopped", 1);
    REQUIRE(!s0.is_null() && !e0.is_null() && !s1.is_null() && !e1.is_null());
    /* Speech began at 500 ms; the turn takes in the 200 ms before it. */
    CHECK(s0["audio_start_ms"].get<int>() >= 280 && s0["audio_start_ms"].get<int>() <= 320);
    CHECK(e0["audio_end_ms"].get<int>() >= 2000 && e0["audio_end_ms"].get<int>() <= 2400);
    CHECK(s1["audio_start_ms"].get<int>() > e0["audio_end_ms"].get<int>());
    CHECK(s0["item_id"] != s1["item_id"]);

    json c0 = r.conn.wait_for("conversation.item.input_audio_transcription.completed", 0);
    json c1 = r.conn.wait_for("conversation.item.input_audio_transcription.completed", 1);
    REQUIRE(!c0.is_null() && !c1.is_null());
    CHECK(c0["item_id"] == s0["item_id"]);
    CHECK(c1["item_id"] == s1["item_id"]);
    CHECK(!c0["transcript"].get<std::string>().empty());
    CHECK_EQ(r.deltas(c0["item_id"].get<std::string>()), c0["transcript"].get<std::string>());
    CHECK_EQ(r.deltas(c1["item_id"].get<std::string>()), c1["transcript"].get<std::string>());
    json k1 = r.conn.wait_for("input_audio_buffer.committed", 1);
    REQUIRE(!k1.is_null());
    CHECK(k1["previous_item_id"] == s0["item_id"]);
}

/* A turn as long as one pass is committed at a pause inside it, without being asked, and the rest
 * goes on as the next turn: a pass here is three seconds, and the quiet stretch is at two. Words
 * settled before the cut are the first turn's alone -- nothing settles in the stretch a cut may
 * land in, so no word heard after the cut has already gone out as the first turn's. */
TEST(a_turn_as_long_as_one_pass_is_cut_at_a_pause) {
    RealtimeRig r(/*max_clip=*/16000 * 3);
    r.conn.client(session_update(16000));
    REQUIRE(!r.conn.wait_for("session.updated").is_null());
    int64_t phase = 0;
    r.speak(16000, 2000, true, &phase);
    r.speak(16000, 100, false, &phase);
    r.speak(16000, 1400, true, &phase);
    json k = r.conn.wait_for("input_audio_buffer.committed");
    REQUIRE(!k.is_null());
    json commit;
    commit["type"] = "input_audio_buffer.commit";
    r.conn.client(commit);
    json c0 = r.conn.wait_for("conversation.item.input_audio_transcription.completed", 0);
    json c1 = r.conn.wait_for("conversation.item.input_audio_transcription.completed", 1);
    REQUIRE(!c0.is_null() && !c1.is_null());
    /* Cut at 2.05 s: four words heard before it, two after. */
    CHECK_EQ(c0["transcript"].get<std::string>(), heard_words(4));
    CHECK_EQ(c1["transcript"].get<std::string>(), heard_words(2));
    CHECK_EQ(r.deltas(c0["item_id"].get<std::string>()), heard_words(4));
    CHECK(!r.deltas(c0["item_id"].get<std::string>()).empty());
    std::lock_guard<std::mutex> lk(r.rig.audio.mu);
    for (int64_t n : r.rig.audio.clips) CHECK(n <= 16000 * 3);
}

/* The audio is at the client's rate, 24 kHz unless it says otherwise, and reaches the model at the
 * model's; and the beta's session.update spelling is answered in the beta's. */
TEST(realtime_audio_is_resampled_to_the_models_rate) {
    RealtimeRig r;
    json beta;
    beta["type"] = "transcription_session.update";
    beta["session"]["input_audio_format"] = "pcm16";
    beta["session"]["input_audio_transcription"]["language"] = "en";
    r.conn.client(beta);
    json up = r.conn.wait_for("transcription_session.updated");
    REQUIRE(!up.is_null());
    CHECK_EQ(up["session"]["input_audio_transcription"]["language"].get<std::string>(),
             std::string("en"));
    int64_t phase = 0;
    r.conn.client_binary(pcm16(24000, 1500, true, &phase));   /* a binary frame: the same bytes */
    json commit;
    commit["type"] = "input_audio_buffer.commit";
    r.conn.client(commit);
    json done = r.conn.wait_for("conversation.item.input_audio_transcription.completed");
    REQUIRE(!done.is_null());
    CHECK_EQ(done["transcript"].get<std::string>(), heard_words(3));
    std::lock_guard<std::mutex> lk(r.rig.audio.mu);
    REQUIRE(!r.rig.audio.clips.empty());
    CHECK(r.rig.audio.clips.back() >= 23990 && r.rig.audio.clips.back() <= 24000);
}

/* What a session cannot do is refused by name, as an error event naming the parameter -- and the
 * session goes on. */
TEST(a_realtime_session_refuses_what_it_cannot_do_and_goes_on) {
    RealtimeRig r;
    REQUIRE(!r.conn.wait_for("session.created").is_null());
    json vad;
    vad["type"] = "semantic_vad";
    json bad = session_update(16000, vad);
    bad["event_id"] = "client-1";
    r.conn.client(bad);
    json e = r.conn.wait_for("error", 0);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["param"].get<std::string>(), std::string("turn_detection.type"));
    CHECK_EQ(e["error"]["event_id"].get<std::string>(), std::string("client-1"));
    r.conn.client(session_update(16000, json(), "xx"));
    e = r.conn.wait_for("error", 1);
    REQUIRE(!e.is_null());
    CHECK(has(e["error"]["message"].get<std::string>(), "en, zh"));
    r.conn.client(session_update(44000));
    e = r.conn.wait_for("error", 2);
    REQUIRE(!e.is_null());
    CHECK(has(e["error"]["message"].get<std::string>(), "24000"));
    json commit;
    commit["type"] = "input_audio_buffer.commit";
    r.conn.client(commit);
    e = r.conn.wait_for("error", 3);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("input_audio_buffer_commit_empty"));
    json junk;
    junk["type"] = "input_audio_buffer.append";
    junk["audio"] = "%%%";
    r.conn.client(junk);
    REQUIRE(!r.conn.wait_for("error", 4).is_null());
    json other;
    other["type"] = "response.create";
    r.conn.client(other);
    e = r.conn.wait_for("error", 5);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("unknown_event"));
    {
        std::lock_guard<std::mutex> lk(r.conn.m);
        r.conn.in.emplace_back("not json", false);
        r.conn.cv.notify_all();
    }
    REQUIRE(!r.conn.wait_for("error", 6).is_null());
    /* Nothing refused changed the session: it still detects the language. */
    r.conn.client(session_update(16000));
    json up = r.conn.wait_for("session.updated");
    REQUIRE(!up.is_null());
    CHECK(up["session"]["audio"]["input"]["transcription"]["language"].is_null());
    CHECK(!r.conn.closed);

    /* A plain GET of the path is told it is a socket. */
    CHECK_EQ(r.rig.srv->router().call("GET", "/v1/realtime").status, 426);
    SpeechRig deaf(/*transcribes=*/false);
    FakeConn c;
    HttpRequest q;
    q.path = "/v1/realtime";
    (*deaf.srv->router().socket_for("/v1/realtime"))(q, c);
    CHECK(c.closed);
    CHECK(has(c.out.front(), "model_not_supported"));
}

/* A session takes one admission place for its life, so sessions count against the queue like
 * requests: with a queue of one, a second session is refused by name and closed, and the place
 * comes back when the first ends. */
TEST(a_realtime_session_holds_an_admission_place) {
    {
        RealtimeRig r(16000 * 30, /*queue_depth=*/1);
        REQUIRE(!r.conn.wait_for("session.created").is_null());
        CHECK_EQ(r.rig.srv->in_flight(), (int64_t)1);
        FakeConn c;
        HttpRequest q;
        q.path = "/v1/realtime";
        (*r.rig.srv->router().socket_for("/v1/realtime"))(q, c);
        CHECK(c.closed);
        CHECK_EQ(c.close_code, 1013);
        REQUIRE(!c.out.empty());
        CHECK(has(c.out.front(), "rate_limit_exceeded"));
        r.conn.hangup();
        r.route.join();
        CHECK_EQ(r.rig.srv->in_flight(), (int64_t)0);
    }
}

/* A session that commits turns faster than they are transcribed is closed with an error once it
 * is too far behind, rather than holding every turn's audio. The passes here never finish. */
TEST(a_realtime_session_too_far_behind_is_closed_with_an_error) {
    RealtimeRig r;
    r.rig.sched.hang = true;
    REQUIRE(!r.conn.wait_for("session.created").is_null());
    r.conn.client(session_update(16000));
    REQUIRE(!r.conn.wait_for("session.updated").is_null());
    json commit;
    commit["type"] = "input_audio_buffer.commit";
    int64_t phase = 0;
    for (int i = 0; i < 70; ++i) {
        r.conn.client(append_event(pcm16(16000, 1000, true, &phase)));
        r.conn.client(commit);
    }
    json e = r.conn.wait_for("error", 0, 5000);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("buffer_overflow"));
    CHECK(has(e["error"]["message"].get<std::string>(), "committed turns"));
    CHECK(eventually([&] {
        std::lock_guard<std::mutex> lk(r.conn.m);
        return r.conn.closed;
    }));
    CHECK_EQ(r.conn.close_code, 1008);
}

/* The client's messages are bounded while the session cannot take them in: with the session's
 * thread held in a send, the reader stops at the queue's bound, and the session then says so
 * before it ends, though the reader has already left. */
TEST(a_realtime_session_that_overflows_says_so_before_it_ends) {
    RealtimeRig r;
    REQUIRE(!r.conn.wait_for("session.created").is_null());
    r.conn.set_hold(true);
    r.conn.client(session_update(16000));   /* its answer blocks the session's thread */
    REQUIRE(eventually([&] {
        std::lock_guard<std::mutex> lk(r.conn.m);
        return r.conn.held == 1;
    }));
    /* Sixteen 1 MiB messages fill the queue; the reader stops at the seventeenth. */
    const std::string chunk(1 << 20, '\0');
    for (int i = 0; i < 20; ++i) r.conn.client_binary(chunk);
    REQUIRE(eventually([&] {
        std::lock_guard<std::mutex> lk(r.conn.m);
        return r.conn.in.size() == 3;
    }));
    r.conn.set_hold(false);
    json e = r.conn.wait_for("error", 0, 5000);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("buffer_overflow"));
}

/* An exception in the session's thread ends the session with an error event and a 1011 close;
 * the server goes on. */
TEST(a_fault_in_a_realtime_session_ends_the_session_not_the_server) {
    RealtimeRig r;
    r.rig.audio.throw_on_prepare = true;
    REQUIRE(!r.conn.wait_for("session.created").is_null());
    r.conn.client(session_update(16000));
    int64_t phase = 0;
    r.conn.client(append_event(pcm16(16000, 1000, true, &phase)));
    json e = r.conn.wait_for("error", 0, 5000);
    REQUIRE(!e.is_null());
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("internal_error"));
    CHECK(eventually([&] {
        std::lock_guard<std::mutex> lk(r.conn.m);
        return r.conn.closed;
    }));
    CHECK_EQ(r.conn.close_code, 1011);
    r.conn.hangup();
    r.route.join();
    CHECK_EQ(r.rig.srv->in_flight(), (int64_t)0);
    CHECK_EQ(r.rig.srv->router().call("GET", "/health").status, 200);
}

RAD_TEST_MAIN()
