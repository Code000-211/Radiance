/* tests/transport_test.cpp -- the server on a real socket.
 *
 * Every other server test dispatches in process, past the transport. These bind 127.0.0.1 on port
 * 0 and speak HTTP/1.1 by hand, for what only a socket shows: the worker pool's bounds, the API key
 * gate, a client that hangs up, keep-alive, chunked bodies, and the status the listener answers a
 * wrong method or an unknown path with.
 */
#include "server_fakes.h"

#include "cpp-httplib/httplib.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>

/* A loopback connection that gives up on a read after `timeout_s`, or -1. */
static int dial(int port, int timeout_s) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    timeval tv{};
    tv.tv_sec = timeout_s;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, (const sockaddr*)&a, sizeof a) < 0) { ::close(fd); return -1; }
    return fd;
}

static bool send_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

/* The status line of the response on `fd`, or "" if none arrived before the read timeout. */
static std::string status_line(int fd) {
    std::string got;
    char buf[512];
    while (got.find("\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        got.append(buf, (size_t)n);
    }
    const size_t e = got.find("\r\n");
    return e == std::string::npos ? std::string() : got.substr(0, e);
}

/* A Server on a real socket, torn down in the order the engine uses: connections first, so every
 * worker they hold comes back, then the listener, then the scheduler's own threads. */
struct LiveServer {
    FakeScheduler           sched;
    FakeTokenizer           tok;
    std::unique_ptr<Server> srv;
    std::thread             serving;
    std::vector<int>        fds;

    /* Generations never finish on their own; each one holds its worker until its peer leaves. */
    LiveServer(int n_threads, int64_t max_seqs,
               const std::function<void(ServerOptions&)>& tweak = nullptr) {
        sched.reply = "abcdefgh";
        sched.hang = true;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        ServerOptions o;
        o.host = "127.0.0.1";
        o.port = 0;
        o.n_threads = n_threads;
        o.max_seqs = max_seqs;
        o.default_max_tokens = 32;
        if (tweak) tweak(o);
        srv.reset(new Server(d, o));
        serving = std::thread([this] { srv->run(); });
        for (int i = 0; i < 500 && srv->port() == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ~LiveServer() {
        for (int fd : fds) ::close(fd);
        srv->stop();
        serving.join();
        sched.join_all();
        srv.reset();
        sched.check_reaped();
    }
    int open(const std::string& send, int timeout_s = 5) {
        const int fd = dial(srv->port(), timeout_s);
        if (fd < 0) return -1;
        fds.push_back(fd);
        return send_all(fd, send) ? fd : -1;
    }
};

/* A stop requested before the listener runs is not lost.
 *
 * The engine stops the server from its watcher, which may run while the transport is still
 * registering its routes; httplib's stop() only closes a listening socket and is a no-op before
 * its accept loop. A transport that is asked to stop first (the same call, before run()) must
 * return instead of serving a listener nothing can stop. Nothing here is timed: a stop that is
 * dropped leaves run() blocked in the accept loop forever, and the bounded wait below is only
 * how that failure is reported. */
TEST(a_stop_before_the_listener_runs_is_remembered) {
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.n_threads = 2;
    o.max_seqs = 2;
    o.default_max_tokens = 32;
    Server srv(d, o);

    srv.stop();
    std::promise<int> ran;
    std::future<int> done = ran.get_future();
    std::thread run([&] { ran.set_value(srv.run()); });
    if (done.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        std::fprintf(stderr, "transport_test: run() did not return after a stop requested before "
                             "the listener ran: the stop was dropped and the accept loop is "
                             "waiting for one\n");
        std::fflush(stderr);
        std::_Exit(1);
    }
    run.join();
    CHECK_EQ(done.get(), RAD_OK);
}

/* The transport calls this on the serve thread after binding and registering the routes, and
 * before httplib's accept loop sets is_running_. The case below arms it once, so it stops exactly
 * in that window; every other serve in this binary sees null. */
static Server* g_stop_before_listen = nullptr;
extern "C" void rad_test_before_listen() {
    if (Server* s = g_stop_before_listen) {
        g_stop_before_listen = nullptr;
        s->stop();
    }
}

/* A stop that arrives after the routes are registered and the socket is bound, but before httplib
 * marks the listener running, is delivered rather than dropped.
 *
 * This is the window the engine's watcher races: the socket is bound, the routes are registered,
 * and httplib's stop() still sees is_running_ == false, so it closes nothing. The seam above stops
 * exactly there, once, on the serve thread. A stop that only reaches a listener already marked
 * running leaves run() blocked in the accept loop; the bounded wait below reports that as the
 * failure it is. */
TEST(a_stop_before_the_accept_loop_starts_is_delivered) {
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.n_threads = 2;
    o.max_seqs = 2;
    o.default_max_tokens = 32;
    Server srv(d, o);

    g_stop_before_listen = &srv;
    std::promise<int> ran;
    std::future<int> done = ran.get_future();
    std::thread run([&] { ran.set_value(srv.run()); });
    if (done.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        std::fprintf(stderr, "transport_test: run() did not return after a stop requested between "
                             "the bind and the accept loop: the stop was dropped and the listener "
                             "is waiting for another one\n");
        std::fflush(stderr);
        std::_Exit(1);
    }
    run.join();
    g_stop_before_listen = nullptr;  /* the hook has run; never arm a later case */
    CHECK_EQ(done.get(), RAD_OK);
}

/* The pool outgrows the admission ceiling. A generation holds its connection's worker while it
 * runs, so a pool no larger than the ceiling fills before admission refuses anyone: the 429 never
 * comes, and a probe queues behind generations. Here the ceiling is 8 and the resident pool 4.
 * Every admitted request is running, forty connections sit on half-sent headers, and the ninth
 * request is still told 429 while /health still answers. */
TEST(a_probe_and_a_429_are_answered_while_every_admitted_request_holds_a_worker) {
    LiveServer s(/*n_threads=*/4, /*max_seqs=*/1);
    REQUIRE(s.srv->port() > 0);

    const std::string gen = "POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
                            "Content-Type: application/json\r\nContent-Length: 15\r\n\r\n"
                            "{\"prompt\":\"hi\"}";
    for (int i = 0; i < 8; ++i) CHECK(s.open(gen) >= 0);
    for (int i = 0; i < 500 && s.srv->in_flight() < 8; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK_EQ(s.srv->in_flight(), (int64_t)8);

    for (int i = 0; i < 40; ++i) CHECK(s.open("GET /health HTTP/1.1\r\nHost: t\r\n") >= 0);

    const int over = s.open(gen);
    REQUIRE(over >= 0);
    CHECK(has(status_line(over), " 429 "));

    const int probe = s.open("GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    REQUIRE(probe >= 0);
    CHECK(has(status_line(probe), " 200 "));
}

/* The response head on `fd`: everything up to the blank line, or what arrived before the read
 * timeout. */
static std::string response_head(int fd) {
    std::string got;
    char buf[512];
    while (got.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        got.append(buf, (size_t)n);
    }
    return got.substr(0, got.find("\r\n\r\n"));
}

/* The transport settings reach the listener: CORS headers are on unless turned off, and a body
 * past --max-body-mib is refused before it is read. */
TEST(the_transport_settings_reach_the_listener) {
    const std::string probe = "GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n";
    {
        LiveServer s(4, 1);
        REQUIRE(s.srv->port() > 0);
        const int fd = s.open(probe);
        REQUIRE(fd >= 0);
        CHECK(has(response_head(fd), "Access-Control-Allow-Origin: *"));
    }
    {
        LiveServer s(4, 1, [](ServerOptions& o) {
            o.cors = false;
            o.max_body_bytes = 64;
        });
        REQUIRE(s.srv->port() > 0);
        const int fd = s.open(probe);
        REQUIRE(fd >= 0);
        const std::string head = response_head(fd);
        CHECK(has(head, " 200 "));
        CHECK(!has(head, "Access-Control-Allow-Origin"));

        const std::string big = std::string(R"({"prompt":")") + std::string(100, 'x') + "\"}";
        const int b = s.open("POST /v1/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
                             "Content-Type: application/json\r\nContent-Length: " +
                             std::to_string(big.size()) + "\r\n\r\n" + big);
        REQUIRE(b >= 0);
        CHECK(has(status_line(b), " 413 "));
    }
}

/* ================================================================== whole responses */

namespace {

/* One response read off the socket: the status, the head, and the body after Content-Length or
 * chunked framing (HEAD carries none). Status 0 when nothing complete arrived before the read
 * timeout. */
struct Reply {
    int status = 0;
    std::string head, body;
    std::string header(const std::string& name) const {
        std::string h = head, n = "\r\n" + name + ":";
        for (char& c : h) c = (char)std::tolower((unsigned char)c);
        for (char& c : n) c = (char)std::tolower((unsigned char)c);
        const size_t at = h.find(n);
        if (at == std::string::npos) return std::string();
        size_t b = at + n.size();
        while (b < head.size() && head[b] == ' ') ++b;
        return head.substr(b, head.find("\r\n", b) - b);
    }
};

Reply read_reply(int fd, bool head_only = false) {
    Reply r;
    std::string buf;
    char tmp[4096];
    auto fill = [&] {
        const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        return true;
    };
    while (buf.find("\r\n\r\n") == std::string::npos)
        if (!fill()) return r;
    const size_t he = buf.find("\r\n\r\n");
    r.head = buf.substr(0, he);
    buf.erase(0, he + 4);
    const int status = std::atoi(r.head.c_str() + r.head.find(' ') + 1);
    if (head_only) { r.status = status; return r; }
    if (r.header("Transfer-Encoding") == "chunked") {
        for (;;) {
            while (buf.find("\r\n") == std::string::npos)
                if (!fill()) return r;
            const size_t n = std::strtoul(buf.c_str(), nullptr, 16);
            buf.erase(0, buf.find("\r\n") + 2);
            while (buf.size() < n + 2)
                if (!fill()) return r;
            r.body.append(buf, 0, n);
            buf.erase(0, n + 2);
            if (n == 0) break;
        }
    } else {
        const size_t n = std::strtoul(r.header("Content-Length").c_str(), nullptr, 10);
        while (buf.size() < n)
            if (!fill()) return r;
        r.body = buf.substr(0, n);
    }
    r.status = status;
    return r;
}

std::string get(const std::string& path, const std::string& headers = "") {
    return "GET " + path + " HTTP/1.1\r\nHost: t\r\n" + headers + "\r\n";
}

std::string post(const std::string& path, const std::string& body, const std::string& headers = "") {
    return "POST " + path + " HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n" + headers +
           "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
}

/* `req` sent on a new connection, and the response read off it. */
Reply ask(LiveServer& s, const std::string& req, bool head_only = false) {
    const int fd = s.open(req);
    return fd < 0 ? Reply{} : read_reply(fd, head_only);
}

}  /* namespace */

/* ================================================================== the API key */

/* With --api-key every route but the probes and the page asks for it, as `Authorization: Bearer`
 * (OpenAI's; the scheme is case-insensitive, RFC 9110) or `X-Api-Key` (Anthropic's), under any
 * capitalisation of the header names. A refusal is a 401 with an error object and the CORS
 * headers a browser needs to read it, and arrives before the body is read: a client sending a
 * large body without the key is answered at once. A preflight needs no key. */
TEST(the_api_key_gate_holds_every_route_but_the_probes) {
    LiveServer s(4, 1, [](ServerOptions& o) { o.api_key = "k"; });
    REQUIRE(s.srv->port() > 0);
    s.sched.hang = false;
    s.sched.reply = "ok";
    auto ask = [&](const std::string& req) { return ::ask(s, req); };
    for (const char* h : { "", "Authorization: Bearer wrong\r\n", "X-Api-Key: wrong\r\n",
                           "Authorization: k\r\n", "Authorization: Basic k\r\n" }) {
        const Reply r = ask(get("/v1/models", h));
        CHECK_EQ(r.status, 401);
        const json e = json::parse(r.body, nullptr, false);
        CHECK(e.is_object() && e["error"].value("type", "") == "authentication_error");
        CHECK_EQ(r.header("Access-Control-Allow-Origin"), std::string("*"));
    }
    for (const char* h : { "Authorization: Bearer k\r\n", "X-Api-Key: k\r\n",
                           "authorization: Bearer k\r\n", "x-api-key: k\r\n",
                           "Authorization: bearer k\r\n", "Authorization: BEARER k\r\n" }) {
        const Reply r = ask(get("/v1/models", h));
        CHECK_EQ(r.status, 200);
        if (r.status != 200) fprintf(stderr, "    %s-> %d\n", h, r.status);
    }
    for (const char* p : { "/health", "/v1/health", "/ping", "/", "/dashboard" })
        CHECK_EQ(ask(get(p)).status, 200);
    CHECK_EQ(ask(get("/metrics")).status, 401);
    CHECK_EQ(ask(get("/stats")).status, 401);
    CHECK_EQ(ask(post("/tokenize", R"({"prompt":"a"})")).status, 401);
    CHECK_EQ(ask(post("/tokenize", R"({"prompt":"a"})", "Authorization: Bearer k\r\n")).status, 200);
    CHECK_EQ(ask("OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                 "Access-Control-Request-Headers: authorization\r\n\r\n").status, 204);
    /* Headers that announce 100 MB and no body: the 401 comes without waiting for it. */
    const Reply big = ask("POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
                          "Content-Length: 100000000\r\n\r\n");
    CHECK_EQ(big.status, 401);
}

/* A refusal the transport writes itself on an Anthropic path (the key gate's 401, a body over
 * the limit, a wrong method, an unknown path below /v1/messages) is in Anthropic's envelope, as
 * the handlers' refusals there are; the same refusal elsewhere stays OpenAI's. */
TEST(the_transports_refusals_on_anthropic_paths_are_in_anthropics_envelope) {
    LiveServer s(4, 1, [](ServerOptions& o) { o.api_key = "k"; o.max_body_bytes = 64; });
    REQUIRE(s.srv->port() > 0);
    const std::string key = "X-Api-Key: k\r\n";
    auto anth = [](const Reply& r, const char* type) {
        const json e = json::parse(r.body, nullptr, false);
        const bool ok = e.is_object() && e.value("type", "") == "error" &&
                        e["error"].is_object() && e["error"].value("type", "") == type &&
                        !e["error"].value("message", "").empty();
        if (!ok) fprintf(stderr, "    %d %s\n", r.status, r.body.c_str());
        return ok;
    };
    const std::string msg = R"({"model":"m","messages":[{"role":"user","content":"hi"}]})";
    for (const char* p : { "/v1/messages", "/v1/messages/count_tokens" }) {
        const Reply r = ::ask(s, post(p, msg, "X-Api-Key: wrong\r\n"));
        CHECK_EQ(r.status, 401);
        CHECK(anth(r, "authentication_error"));
    }
    const Reply big = ::ask(s, post("/v1/messages", std::string(100, ' ') + msg, key));
    CHECK_EQ(big.status, 413);
    CHECK(anth(big, "request_too_large"));
    const Reply wrong = ::ask(s, get("/v1/messages", key));
    CHECK_EQ(wrong.status, 405);
    CHECK(anth(wrong, "invalid_request_error"));
    const Reply none = ::ask(s, post("/v1/messages/batches", "{}", key));
    CHECK_EQ(none.status, 404);
    CHECK(anth(none, "not_found_error"));
    /* The same refusal on another path is OpenAI's. */
    const Reply oai = ::ask(s, post("/v1/chat/completions", msg, "X-Api-Key: wrong\r\n"));
    CHECK_EQ(oai.status, 401);
    const json e = json::parse(oai.body, nullptr, false);
    CHECK(e.is_object() && !e.contains("type") &&
          e["error"].value("type", "") == "authentication_error");
}

/* ================================================================== a client that leaves */

/* A client that closes its socket mid-stream, or while waiting for a buffered answer, cancels its
 * request at the scheduler and gives its admission place back: the binding from a dead socket to
 * a cancellation is the transport's, which no in-process test reaches. Both count as
 * disconnects, and the server admits new work after. */
TEST(a_client_closing_its_socket_cancels_its_request) {
    LiveServer s(4, 2);
    REQUIRE(s.srv->port() > 0);
    const int st = s.open(post("/v1/completions", R"({"prompt":"hi","stream":true})"));
    REQUIRE(st >= 0);
    {
        /* The head and the first token's chunk, then the client leaves. */
        std::string got;
        char buf[1024];
        while (got.find("data: ") == std::string::npos) {
            const ssize_t n = ::recv(st, buf, sizeof buf, 0);
            if (n <= 0) break;
            got.append(buf, (size_t)n);
        }
        CHECK(has(got, " 200 "));
    }
    ::close(st);
    s.fds.erase(std::find(s.fds.begin(), s.fds.end(), st));
    CHECK(eventually([&] { return s.srv->in_flight() == 0; }));
    CHECK(s.sched.saw_cancel());

    const int bf = s.open(post("/v1/completions", R"({"prompt":"hi"})"));
    REQUIRE(bf >= 0);
    CHECK(eventually([&] { return s.srv->in_flight() == 1; }));
    ::close(bf);
    s.fds.erase(std::find(s.fds.begin(), s.fds.end(), bf));
    CHECK(eventually([&] { return s.srv->in_flight() == 0; }));
    {
        std::lock_guard<std::mutex> lk(s.sched.mu);
        CHECK_EQ(s.sched.cancelled.size(), (size_t)2);
    }
    const std::string m = s.srv->router().call("GET", "/metrics").body;
    CHECK(has(m, "radiance:client_disconnect_total{model_name=\"radiance\",engine=\"0\"} 2\n"));

    s.sched.hang = false;
    s.sched.reply = "ok";
    const Reply r = ask(s, post("/v1/completions", R"({"prompt":"hi"})"));
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["choices"][0].value("text", ""), std::string("ok"));
}

/* ================================================================== the wire itself */

/* One connection carries a streamed completion, then a probe, then a buffered completion: the
 * stream's end leaves the connection usable. */
TEST(a_keep_alive_connection_serves_a_stream_and_what_follows) {
    LiveServer s(4, 1);
    REQUIRE(s.srv->port() > 0);
    s.sched.hang = false;
    s.sched.reply = "ok";
    const int fd = s.open(post("/v1/completions", R"({"prompt":"hi","stream":true})"));
    REQUIRE(fd >= 0);
    const Reply a = read_reply(fd);
    CHECK_EQ(a.status, 200);
    CHECK(has(a.body, "data: [DONE]\n\n"));
    REQUIRE(send_all(fd, get("/health")));
    CHECK_EQ(read_reply(fd).status, 200);
    REQUIRE(send_all(fd, post("/v1/completions", R"({"prompt":"hi"})")));
    const Reply c = read_reply(fd);
    CHECK_EQ(c.status, 200);
    CHECK(json::accept(c.body));
}

/* A body sent chunked (no Content-Length) is read whole, and one past --max-body-mib is a 413
 * however it is framed. */
TEST(a_chunked_request_body_is_read_and_bounded) {
    LiveServer s(4, 1, [](ServerOptions& o) { o.max_body_bytes = 64; });
    REQUIRE(s.srv->port() > 0);
    auto chunked = [](const std::string& body) {
        std::string req = "POST /tokenize HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
                          "Transfer-Encoding: chunked\r\n\r\n";
        for (size_t at = 0; at < body.size(); at += 7) {
            const std::string piece = body.substr(at, 7);
            char len[16];
            std::snprintf(len, sizeof len, "%zx\r\n", piece.size());
            req += len + piece + "\r\n";
        }
        return req + "0\r\n\r\n";
    };
    const Reply r = ask(s, chunked(R"({"prompt":"AB"})"));
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"count\":2"));
    const int big = s.open(chunked(R"({"prompt":")" + std::string(100, 'x') + "\"}"));
    REQUIRE(big >= 0);
    CHECK(has(status_line(big), " 413 "));
}

/* A wrong method on a route that exists is 405 over the socket as in process (http.h), whatever
 * the method, with an error object; an unknown path is a 404 with one; HEAD is GET without the
 * body, so a probe that sends it is answered. */
TEST(the_listener_answers_a_wrong_method_405_and_head_as_get) {
    LiveServer s(4, 1);
    REQUIRE(s.srv->port() > 0);
    auto ask = [&](const std::string& req, bool head_only = false) { return ::ask(s, req, head_only); };
    struct Case { const char* method; const char* path; int status; };
    for (const Case& c : { Case{ "GET", "/v1/chat/completions", 405 },
                           Case{ "PUT", "/v1/chat/completions", 405 },
                           Case{ "DELETE", "/health", 405 },
                           Case{ "PATCH", "/v1/models", 405 },
                           Case{ "GET", "/v1/nope", 404 } }) {
        const Reply r = ask(std::string(c.method) + " " + c.path + " HTTP/1.1\r\nHost: t\r\n"
                            "Content-Length: 0\r\n\r\n");
        CHECK_EQ(r.status, c.status);
        if (r.status != c.status) fprintf(stderr, "    %s %s -> %d\n", c.method, c.path, r.status);
        const json e = json::parse(r.body, nullptr, false);
        CHECK(e.is_object() && e.contains("error"));
    }
    CHECK_EQ(ask(post("/v1/models", "{}")).status, 405);
    const Reply head = ask("HEAD /health HTTP/1.1\r\nHost: t\r\n\r\n", true);
    CHECK_EQ(head.status, 200);
}

/* The listener's 404 names the path in a body a strict client can parse: valid UTF-8 whatever
 * bytes the decoded path held. */
TEST(the_listeners_404_is_valid_json_whatever_the_path_held) {
    LiveServer s(4, 1);
    REQUIRE(s.srv->port() > 0);
    const Reply odd = ask(s, get("/%FF%C3"));
    CHECK_EQ(odd.status, 404);
    CHECK(json::accept(odd.body));
}

/* A served name holding a slash is retrieved by its id over the socket, sent with the slash or
 * as %2F. */
TEST(the_listener_retrieves_a_model_whose_name_holds_a_slash) {
    LiveServer s(4, 1, [](ServerOptions& o) { o.model_id = "org/model"; });
    REQUIRE(s.srv->port() > 0);
    for (const char* p : { "/v1/models/org/model", "/v1/models/org%2Fmodel" }) {
        const Reply r = ask(s, get(p));
        CHECK_EQ(r.status, 200);
        CHECK_EQ(json::parse(r.body, nullptr, false).value("id", ""), std::string("org/model"));
    }
}

/* An upload over a real socket reaches the route as form parts: the transport reads a
 * multipart/form-data body part by part rather than discarding it. */
TEST(a_multipart_upload_reaches_the_transcription_route_over_a_socket) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeAudio audio;
    SpeechFormat fmt;
    fmt.transcribe_prompt = "{audio}{language}";
    fmt.transcript_open = "<t>";
    fmt.languages = { { "en", "English" } };
    sched.reply = "<t>socket words";
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.mm = &audio;
    d.speech = &fmt;
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.n_threads = 2;
    o.max_seqs = 2;
    std::unique_ptr<Server> srv(new Server(d, o));
    std::thread serving([&] { srv->run(); });
    for (int i = 0; i < 500 && srv->port() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE(srv->port() > 0);

    const std::string b = "BoUnDaRy";
    std::string body = "--" + b + "\r\nContent-Disposition: form-data; name=\"file\"; "
                       "filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + tone(4000) +
                       "\r\n--" + b + "\r\nContent-Disposition: form-data; "
                       "name=\"response_format\"\r\n\r\ntext\r\n--" + b + "--\r\n";
    const std::string req = "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: t\r\n"
                            "Content-Type: multipart/form-data; boundary=" + b + "\r\n"
                            "Content-Length: " + std::to_string(body.size()) +
                            "\r\nConnection: close\r\n\r\n" + body;
    const int fd = dial(srv->port(), 5);
    REQUIRE(fd >= 0);
    CHECK(send_all(fd, req));
    std::string got;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        got.append(buf, (size_t)n);
    }
    ::close(fd);
    CHECK(has(got, " 200 "));
    CHECK(has(got, "socket words\n"));
    CHECK_EQ(audio.clips.size(), (size_t)1);
    if (!audio.clips.empty()) CHECK_EQ(audio.clips[0], (int64_t)4000);

    srv->stop();
    serving.join();
    sched.join_all();
    srv.reset();
}

/* Over a real socket: the upgrade, the key a browser carries as a subprotocol, and a session from
 * the first event to the completed transcript. */
TEST(a_realtime_session_runs_over_a_websocket) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeAudio audio;
    audio.max_clip = 16000 * 30;
    audio.per_token = 1600;
    sched.reply_for = asr_reply;
    SpeechFormat fmt;
    fmt.transcribe_prompt = "<s>{context}|{audio}|{language}";
    fmt.language_prompt = "lang {language}<t>";
    fmt.transcript_open = "<t>";
    fmt.language_lead = "lang ";
    fmt.languages = { { "en", "English" } };
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.mm = &audio;
    d.speech = &fmt;
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.n_threads = 2;
    o.max_seqs = 2;
    o.api_key = "sekrit";
    std::unique_ptr<Server> srv(new Server(d, o));
    std::thread serving([&] { srv->run(); });
    for (int i = 0; i < 500 && srv->port() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE(srv->port() > 0);
    const std::string url = "ws://127.0.0.1:" + std::to_string(srv->port()) + "/v1/realtime";

    {
        httplib::ws::WebSocketClient nokey(url);
        CHECK(!nokey.connect());
    }
    httplib::Headers h = {
        { "Sec-WebSocket-Protocol", "realtime, openai-insecure-api-key.sekrit" } };
    httplib::ws::WebSocketClient ws(url, h);
    REQUIRE(ws.connect());
    CHECK_EQ(ws.subprotocol(), std::string("realtime"));
    std::string msg;
    REQUIRE(ws.read(msg) == httplib::ws::Text);
    CHECK_EQ(json::parse(msg)["type"].get<std::string>(), std::string("session.created"));
    CHECK(ws.send(session_update(16000).dump()));
    int64_t phase = 0;
    CHECK(ws.send(append_event(pcm16(16000, 1000, true, &phase)).dump()));
    json commit;
    commit["type"] = "input_audio_buffer.commit";
    CHECK(ws.send(commit.dump()));
    std::string transcript;
    for (int i = 0; i < 100 && transcript.empty(); ++i) {
        if (ws.read(msg) != httplib::ws::Text) break;
        json j = json::parse(msg);
        if (j["type"] == "conversation.item.input_audio_transcription.completed")
            transcript = j["transcript"].get<std::string>();
    }
    CHECK_EQ(transcript, heard_words(2));
    ws.close();

    srv->stop();
    serving.join();
    sched.join_all();
    srv.reset();
}

/* A WebSocket message is bounded: one over 2 MiB closes the connection instead of being read.
 * The client is spoken by hand, since httplib's own client refuses to send such a message. */
TEST(a_realtime_message_over_the_bound_closes_the_socket) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeAudio audio;
    audio.max_clip = 16000 * 30;
    SpeechFormat fmt;
    fmt.transcribe_prompt = "<s>{context}|{audio}|{language}";
    fmt.transcript_open = "<t>";
    fmt.languages = { { "en", "English" } };
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.mm = &audio;
    d.speech = &fmt;
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.n_threads = 2;
    o.max_seqs = 2;
    std::unique_ptr<Server> srv(new Server(d, o));
    std::thread serving([&] { srv->run(); });
    for (int i = 0; i < 500 && srv->port() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE(srv->port() > 0);

    /* A masked text frame of `n` bytes of 'x' (a zero mask leaves the payload as it is). */
    auto frame = [](size_t n) {
        std::string f;
        f.push_back((char)0x81);
        f.push_back((char)(0x80 | 127));
        for (int i = 7; i >= 0; --i) f.push_back((char)((uint64_t)n >> (8 * i)));
        f.append(4, '\0');
        f.append(n, 'x');
        return f;
    };
    auto received = [](int fd, int timeout_ms) {
        std::string got;
        char buf[4096];
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < until) {
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n <= 0) break;
            got.append(buf, (size_t)n);
            if (has(got, "invalid_json")) break;
        }
        return got;
    };
    const int fd = dial(srv->port(), 2);
    REQUIRE(fd >= 0);
    CHECK(send_all(fd, "GET /v1/realtime HTTP/1.1\r\nHost: t\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                       "Sec-WebSocket-Version: 13\r\n\r\n"));
    CHECK(has(response_head(fd), " 101 "));
    /* Under the bound a message that is not JSON is read and answered with an error... */
    CHECK(send_all(fd, frame((size_t)1 << 20)));
    CHECK(has(received(fd, 5000), "invalid_json"));
    /* ...and over it the connection is closed, the message unread. */
    send_all(fd, frame((size_t)3 << 20));
    CHECK(!has(received(fd, 5000), "invalid_json"));
    ::close(fd);

    srv->stop();
    serving.join();
    sched.join_all();
    srv.reset();
}

RAD_TEST_MAIN()
