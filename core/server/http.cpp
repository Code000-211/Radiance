#include "http.h"
#include "rad_internal.h"
#include "text/tokenizer.h"

/* httplib pulls in <thread>, <sys/socket.h> and much else; only this unit includes it. */
#include "cpp-httplib/httplib.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

/* The transport's last moment before httplib's accept loop: the socket is bound and the routes are
 * registered, but is_running_ is not set yet. A stop requested here has no listener to close; the
 * listener must consume it when it starts. The default is the production no-op; a test binary
 * links a strong definition that stops exactly here. */
extern "C" __attribute__((weak)) void rad_test_before_listen() {}

namespace rad {
namespace server {

bool CiLess::operator()(const std::string& a, const std::string& b) const {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](unsigned char x, unsigned char y) { return std::tolower(x) < std::tolower(y); });
}

std::string HttpRequest::header(const std::string& k, const std::string& def) const {
    auto it = headers.find(k);
    return it == headers.end() ? def : it->second;
}

const FormPart* HttpRequest::part(const std::string& k) const {
    for (const FormPart& p : form)
        if (p.name == k) return &p;
    return nullptr;
}

std::string HttpRequest::param(const std::string& k, const std::string& def) const {
    auto it = params.find(k);
    if (it != params.end()) return it->second;
    auto q = query.find(k);
    return q == query.end() ? def : q->second;
}

bool HttpRequest::peer_is_loopback() const {
    if (peer.empty()) return true;
    /* "addr:port", where an IPv6 address carries colons of its own: the port is after the last. */
    const size_t colon = peer.rfind(':');
    std::string addr = colon == std::string::npos ? peer : peer.substr(0, colon);
    if (addr.size() >= 2 && addr.front() == '[' && addr.back() == ']')
        addr = addr.substr(1, addr.size() - 2);
    in_addr v4{};
    if (inet_pton(AF_INET, addr.c_str(), &v4) == 1) return (ntohl(v4.s_addr) >> 24) == 127;
    in6_addr v6{};
    if (inet_pton(AF_INET6, addr.c_str(), &v6) != 1) return false;
    if (IN6_IS_ADDR_LOOPBACK(&v6)) return true;
    return IN6_IS_ADDR_V4MAPPED(&v6) && v6.s6_addr[12] == 127;
}

int idle_workers(int n_threads) {
    const int n = n_threads > 0 ? n_threads : (int)std::thread::hardware_concurrency();
    return std::max(n, 4);
}

HttpResponse HttpResponse::json_body(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.body = std::move(body);
    return r;
}

HttpResponse HttpResponse::text(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.content_type = "text/plain; charset=utf-8";
    r.body = std::move(body);
    return r;
}

/* ------------------------------------------------------------------ target parsing */

/* `+` is a space in a query string and a plus sign in a path: the form encoding
 * (application/x-www-form-urlencoded) applies to a query only. Decoding a path the query's way
 * would turn /v1/models/gpt+4 into a request for "gpt 4". */
std::string url_decode(const std::string& s, bool plus_is_space) {
    std::string out;
    out.reserve(s.size());
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            int h = hex(s[i + 1]), l = hex(s[i + 2]);
            if (h >= 0 && l >= 0) { out.push_back((char)(h * 16 + l)); i += 2; continue; }
        }
        out.push_back(plus_is_space && s[i] == '+' ? ' ' : s[i]);
    }
    return out;
}

void parse_target(const std::string& target, std::string* path,
                  std::map<std::string, std::string>* query) {
    size_t q = target.find('?');
    if (path) *path = url_decode(target.substr(0, q), /*plus_is_space=*/false);
    if (!query || q == std::string::npos) return;
    size_t i = q + 1;
    while (i < target.size()) {
        size_t amp = target.find('&', i);
        if (amp == std::string::npos) amp = target.size();
        std::string kv = target.substr(i, amp - i);
        size_t eq = kv.find('=');
        if (eq == std::string::npos) (*query)[url_decode(kv)] = "";
        else (*query)[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
        i = amp + 1;
    }
}

/* ------------------------------------------------------------------ routing */

static std::vector<std::string> split_path(const std::string& p) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < p.size()) {
        while (i < p.size() && p[i] == '/') ++i;
        size_t j = i;
        while (j < p.size() && p[j] != '/') ++j;
        if (j > i) out.push_back(p.substr(i, j - i));
        i = j;
    }
    return out;
}

void Router::add(std::string method, std::string path, Handler h) {
    routes_.push_back(Route{std::move(method), split_path(path), std::move(h)});
}

void Router::socket(std::string path, SocketHandler h) {
    sockets_.emplace_back(std::move(path), std::move(h));
}

const SocketHandler* Router::socket_for(const std::string& path) const {
    for (const auto& s : sockets_)
        if (s.first == path) return &s.second;
    return nullptr;
}

std::vector<std::string> Router::socket_routes() const {
    std::vector<std::string> out;
    for (const auto& s : sockets_) out.push_back(s.first);
    return out;
}

std::vector<std::string> Router::routes() const {
    std::vector<std::string> out;
    for (auto& r : routes_) {
        std::string p;
        for (auto& s : r.segs) { p += '/'; p += s; }
        out.push_back(r.method + " " + (p.empty() ? "/" : p));
    }
    return out;
}

bool anthropic_path(const std::string& path) {
    return path == "/v1/messages" || path.rfind("/v1/messages/", 0) == 0;
}

const char* anthropic_error_type(int status) {
    return status == 401 ? "authentication_error"
         : status == 403 ? "permission_error"
         : status == 404 ? "not_found_error"
         : status == 413 ? "request_too_large"
         : status == 429 ? "rate_limit_error"
         : status >= 500 ? "api_error"
                         : "invalid_request_error";
}

/* A refusal the transport or the router writes itself. `type` is OpenAI's; on an Anthropic path
 * the status picks Anthropic's instead. Hand-rolled, not nlohmann: this error path must work even
 * if the JSON layer failed. The message can quote a request's decoded path, so bytes that are not
 * UTF-8 become U+FFFD; a JSON body must be UTF-8 for a client to parse it. */
static std::string error_body(const std::string& path, int status, const char* type,
                              const std::string& msg) {
    std::string esc;
    for (char c : utf8_sanitize(msg)) {
        if (c == '"' || c == '\\') { esc.push_back('\\'); esc.push_back(c); }
        else if (c == '\n') esc += "\\n";
        else if ((unsigned char)c < 0x20) esc += "?";
        else esc.push_back(c);
    }
    if (anthropic_path(path))
        return std::string("{\"type\":\"error\",\"error\":{\"type\":\"") +
               anthropic_error_type(status) + "\",\"message\":\"" + esc + "\"}}";
    return std::string("{\"error\":{\"message\":\"") + esc + "\",\"type\":\"" + type +
           "\",\"param\":null,\"code\":null}}";
}

const Router::Route* Router::match(const std::string& method, const std::vector<std::string>& segs,
                                   std::map<std::string, std::string>* caps,
                                   bool* path_matched) const {
    *path_matched = false;
    for (const auto& r : routes_) {
        const bool rest = !r.segs.empty() && r.segs.back()[0] == '*';
        if (rest ? segs.size() < r.segs.size() : segs.size() != r.segs.size()) continue;
        const size_t fixed = rest ? r.segs.size() - 1 : r.segs.size();
        std::map<std::string, std::string> c;
        bool ok = true;
        for (size_t i = 0; i < fixed; ++i) {
            if (!r.segs[i].empty() && r.segs[i][0] == ':') c[r.segs[i].substr(1)] = segs[i];
            else if (r.segs[i] != segs[i]) { ok = false; break; }
        }
        if (!ok) continue;
        if (rest) {
            std::string& v = c[r.segs.back().substr(1)];
            for (size_t i = fixed; i < segs.size(); ++i) v += (i > fixed ? "/" : "") + segs[i];
        }
        *path_matched = true;
        if (r.method != method) continue;
        if (caps) *caps = std::move(c);
        return &r;
    }
    return nullptr;
}

bool Router::allows(const std::string& method, const std::string& path) const {
    bool path_matched = false;
    return match(method, split_path(path), nullptr, &path_matched) != nullptr;
}

HttpResponse Router::dispatch(HttpRequest req) const {
    bool path_matched = false;
    std::map<std::string, std::string> caps;
    if (const Route* r = match(req.method, split_path(req.path), &caps, &path_matched)) {
        req.params.insert(caps.begin(), caps.end());
        return r->h(req);
    }

    if (path_matched)
        return HttpResponse::json_body(
            405, error_body(req.path, 405, "invalid_request_error",
                            req.method + " is not allowed on " + req.path));
    return HttpResponse::json_body(
        404, error_body(req.path, 404, "not_found_error",
                        "no route for " + req.method + " " + req.path));
}

HttpResponse Router::call(const std::string& method, const std::string& path,
                          const std::string& body) const {
    HttpRequest r;
    r.method = method;
    parse_target(path, &r.path, &r.query);
    r.body = body;
    if (!body.empty()) r.headers["Content-Type"] = "application/json";
    return dispatch(std::move(r));
}

/* ------------------------------------------------------------------ the worker pool */

/* A worker for every connection, up to a bound, started as soon as a connection has none.
 *
 * httplib's ThreadPool counts a worker as idle from its wake-up until it takes its job, so two
 * connections arriving in that window leave the second queued with no new worker, possibly for
 * minutes behind a generation. This pool compares the queue with the workers free to take from
 * it, so a connection waits only once the pool is at its bound.
 *
 * `keep` workers stay up; the rest exit after kIdleExit idle. Past `most` workers, `queued`
 * connections may wait (0: any number) and further ones are refused, which httplib answers by
 * closing the socket. */
class ConnectionPool final : public httplib::TaskQueue {
public:
    ConnectionPool(size_t keep, size_t most, size_t queued)
        : keep_(keep), most_(std::max(most, keep)), queued_(queued) {
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i = 0; i < keep_; ++i) start();
    }
    ~ConnectionPool() override { shutdown(); }

    bool enqueue(std::function<void()> fn) override {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) return false;
            if (jobs_.size() >= free_) {       /* no free worker is left for this job */
                const bool started = threads_.size() < most_ && start();
                if (!started && queued_ && jobs_.size() - free_ >= queued_) return false;
            }
            jobs_.push_back(std::move(fn));
        }
        cv_.notify_one();
        return true;
    }

    /* Runs what is queued, then joins every worker. Idempotent: httplib calls it, then the
     * destructor calls it again. */
    void shutdown() override {
        std::list<std::thread> live;
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            live.swap(threads_);
        }
        cv_.notify_all();
        for (auto& t : live) t.join();
        std::lock_guard<std::mutex> lk(m_);
        reap();
    }

private:
    static constexpr std::chrono::seconds kIdleExit{5};

    /* With m_ held. False when no thread can be created; the job then waits. */
    bool start() {
        reap();
        try {
            threads_.emplace_back([this] { work(); });
        } catch (const std::system_error& e) {
            RAD_WARN("server: cannot start an HTTP worker (%s); the connection waits", e.what());
            return false;
        }
        ++free_;
        return true;
    }

    /* With m_ held: join the workers that exited idle. They released m_ before exiting, so the
     * join is short. */
    void reap() {
        for (auto& t : finished_) t.join();
        finished_.clear();
    }

    void work() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            while (jobs_.empty() && !stop_) {
                if (threads_.size() <= keep_) { cv_.wait(lk); continue; }
                if (cv_.wait_for(lk, kIdleExit) == std::cv_status::timeout && jobs_.empty() &&
                    !stop_ && threads_.size() > keep_) {
                    /* Hand this thread's handle to finished_, for the next start() to join. */
                    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
                        if (it->get_id() != std::this_thread::get_id()) continue;
                        finished_.push_back(std::move(*it));
                        threads_.erase(it);
                        break;
                    }
                    --free_;
                    return;
                }
            }
            if (jobs_.empty()) { --free_; return; }    /* stopping, and nothing is left to run */
            std::function<void()> fn = std::move(jobs_.front());
            jobs_.pop_front();
            --free_;
            lk.unlock();
            fn();
            lk.lock();
            ++free_;
        }
    }

    const size_t keep_, most_, queued_;
    std::mutex m_;
    std::condition_variable cv_;
    std::list<std::function<void()>> jobs_;
    std::list<std::thread> threads_;       /* every worker not yet exited */
    std::vector<std::thread> finished_;    /* exited idle, not yet joined */
    size_t free_ = 0;                      /* workers not running a job */
    bool stop_ = false;
};

/* ------------------------------------------------------------------ the httplib binding */

struct HttpTransport::Impl {
    httplib::Server  svr;
    std::atomic<int> port{0};
};

HttpTransport::HttpTransport() : impl_(new Impl) {}
HttpTransport::~HttpTransport() { stop(); }

void HttpTransport::stop() {
    if (!impl_) return;
    impl_->svr.stop();
}

/* 0 until the accept loop runs: a caller that stopped a server it believed up would otherwise
 * wait on a port nothing is listening on. */
int HttpTransport::port() const {
    return impl_ && impl_->svr.is_running() ? impl_->port.load() : 0;
}

/* Compares a credential with the key in time independent of where they differ, so a client
 * cannot find the key a byte at a time by timing refusals. The length is not secret. */
static bool same_secret(const std::string& got, const std::string& key) {
    if (got.size() != key.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < key.size(); ++i) d |= (unsigned char)(got[i] ^ key[i]);
    return d == 0;
}

static void apply(const HttpResponse& r, httplib::Response& res) {
    res.status = r.status;
    for (auto& h : r.headers) res.set_header(h.first, h.second);
}

/* httplib's WebSocket as a WsConn. Sending from another thread is httplib's own practice -- its
 * writes are under a mutex and its heartbeat pings from a thread of its own -- and so is closing
 * from one: its heartbeat closes on a pong timeout while the route's thread is reading. */
class HttplibConn final : public WsConn {
public:
    explicit HttplibConn(httplib::ws::WebSocket& ws) : ws_(ws) {}
    bool read(std::string* msg, bool* binary) override {
        const httplib::ws::ReadResult r = ws_.read(*msg);
        if (r == httplib::ws::Fail) return false;
        *binary = r == httplib::ws::Binary;
        return true;
    }
    bool send(const std::string& text) override { return ws_.send(text); }
    bool send_binary(const std::string& bytes) override {
        return ws_.send(bytes.data(), bytes.size());
    }
    void close(int code, const std::string& reason) override {
        ws_.close((httplib::ws::CloseStatus)code, reason);
    }

private:
    httplib::ws::WebSocket& ws_;
};

static std::string lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

/* The API key a browser's socket carries. A browser cannot set a header on a WebSocket, so
 * OpenAI's browser clients offer it as a subprotocol, `openai-insecure-api-key.<key>`, beside
 * `realtime`; the subprotocol chosen is never that one. Compared in constant time, as the header
 * key is. */
static bool offers_key(const std::string& protocols, const std::string& key) {
    static const std::string kPrefix = "openai-insecure-api-key.";
    size_t i = 0;
    while (i <= protocols.size()) {
        size_t j = protocols.find(',', i);
        if (j == std::string::npos) j = protocols.size();
        size_t a = i, b = j;
        while (a < b && protocols[a] == ' ') ++a;
        while (b > a && protocols[b - 1] == ' ') --b;
        if (b - a >= kPrefix.size() && protocols.compare(a, kPrefix.size(), kPrefix) == 0 &&
            same_secret(protocols.substr(a + kPrefix.size(), b - a - kPrefix.size()), key))
            return true;
        i = j + 1;
    }
    return false;
}

/* httplib's own test for a WebSocket upgrade (RFC 6455 §4.2.1), which it applies before the
 * pre-routing handler and does not export. A request it rejects is routed as a plain GET, so the
 * pre-routing handler must not leave such a request to httplib either. */
static bool is_socket_upgrade(const httplib::Request& q) {
    if (q.method != "GET" || lower_ascii(q.get_header_value("Upgrade")) != "websocket") return false;
    if (lower_ascii(q.get_header_value("Connection")).find("upgrade") == std::string::npos)
        return false;
    const std::string key = q.get_header_value("Sec-WebSocket-Key");
    if (key.size() != 24 || key[22] != '=' || key[23] != '=') return false;
    static const std::string b64 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < 22; ++i)
        if (b64.find(key[i]) == std::string::npos) return false;
    return q.get_header_value("Sec-WebSocket-Version") == "13";
}

int HttpTransport::serve(const Router& router, const TransportOptions& o) {
    auto& svr = impl_->svr;

    /* A request line longer than CPPHTTPLIB_MAX_LINE_LENGTH is closed with no status: httplib's
     * line reader gives up before any handler, error handler or logger runs. The build sets
     * CPPHTTPLIB_REQUEST_URI_MAX_LENGTH to the same limit, so httplib's own 414 never fires. */
    svr.set_read_timeout(o.read_timeout_s, 0);
    svr.set_write_timeout(o.write_timeout_s, 0);
    svr.set_keep_alive_timeout(o.keep_alive_timeout_s);
    svr.set_payload_max_length(o.max_body_bytes);

    const int nt = idle_workers(o.n_threads);
    const size_t most = (size_t)std::max<int64_t>(o.max_threads, nt);
    const size_t queued = (size_t)std::max<int64_t>(o.max_queued, 0);
    svr.new_task_queue = [nt, most, queued] { return new ConnectionPool((size_t)nt, most, queued); };

    /* An exception escaping a handler would close the connection with no body; answer 500 with
     * an error object instead. */
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res,
                                 std::exception_ptr ep) {
        std::string what = "unhandled exception in handler";
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { what = e.what(); }
        catch (...) {}
        RAD_ERR("server: %s", what.c_str());
        res.status = 500;
        res.set_content(error_body(req.path, 500, "internal_server_error", what),
                        "application/json");
    });

    const std::string api_key = o.api_key;
    const bool cors = o.cors;
    /* The headers a cross-origin client sends: the body type, either form of the API key, and the
     * anthropic-version and anthropic-beta headers of Anthropic's clients, which nothing reads. */
    static const char* const kCorsHeaders =
        "Content-Type, Authorization, X-Api-Key, anthropic-version, anthropic-beta";
    static const char* const kCorsMethods = "GET, POST, OPTIONS";

    /* Whether a request carries the API key: `Authorization: Bearer` (OpenAI; the scheme is
     * case-insensitive, RFC 9110) or `X-Api-Key` (Anthropic, also accepted by llama-server). */
    auto key_ok = [api_key](const httplib::Request& hreq) -> bool {
        const std::string auth = hreq.get_header_value("Authorization");
        const size_t sp = auth.find(' ');
        const size_t cred = sp == std::string::npos ? sp : auth.find_first_not_of(' ', sp);
        const bool bearer = cred != std::string::npos && sp == 6 &&
                            std::equal(auth.begin(), auth.begin() + 6, "bearer",
                                       [](char a, char b) {
                                           return std::tolower((unsigned char)a) == b;
                                       });
        return (bearer && same_secret(auth.substr(cred), api_key)) ||
               same_secret(hreq.get_header_value("X-Api-Key"), api_key);
    };

    /* What a request must pass before its body is read, so a refused one never has it buffered. */
    auto gate = [api_key, cors, key_ok](const httplib::Request& hreq,
                                        httplib::Response& hres) -> bool {
        if (cors) {
            hres.set_header("Access-Control-Allow-Origin", "*");
            hres.set_header("Access-Control-Allow-Headers", kCorsHeaders);
            hres.set_header("Access-Control-Allow-Methods", kCorsMethods);
        }
        /* The probe paths and the dashboard page are exempt. A load balancer cannot send a
         * bearer token, and a browser opening the page cannot send a header; the page holds no
         * data and sends the key on every data fetch. Everything else is guarded, including
         * /tokenize, which vLLM leaves open (it only guards /v1). */
        const std::string& p = hreq.path;
        if (!api_key.empty() && p != "/health" && p != "/v1/health" && p != "/ping" && p != "/" &&
            p != "/dashboard") {
            if (!key_ok(hreq)) {
                /* Logged in Server::error_response's format; the gate answers before the router. */
                RAD_INFO("req refused from %s:%d -> 401 authentication_error: invalid api key",
                         hreq.remote_addr.c_str(), hreq.remote_port);
                hres.status = 401;
                hres.set_content(error_body(p, 401, "authentication_error", "invalid api key"),
                                 "application/json");
                return false;
            }
        }
        return true;
    };

    auto bridge = [&router](const httplib::Request& hreq, std::string body,
                            httplib::Response& hres, std::vector<FormPart> form = {}) {
        HttpRequest req;
        /* HEAD is GET without the body, which httplib leaves off the wire. */
        req.method = hreq.method == "HEAD" ? "GET" : hreq.method;
        req.path = hreq.path;
        req.body = std::move(body);
        req.form = std::move(form);
        req.peer = hreq.remote_addr + ":" + std::to_string(hreq.remote_port);
        req.t_received_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        for (auto& h : hreq.headers) req.headers[h.first] = h.second;
        for (auto& p : hreq.params) req.query[p.first] = p.second;
        req.closed = [&hreq] { return hreq.is_connection_closed(); };

        HttpResponse res = router.dispatch(std::move(req));
        apply(res, hres);

        if (!res.streaming()) {
            hres.set_content(std::move(res.body), res.content_type);
            if (res.on_close) res.on_close(true);
            return;
        }

        /* Streaming. The shared_ptr keeps the handler's state alive, because httplib calls the
         * provider after this function returns. The releaser reports whether the peer went away
         * and is where cancellation hooks in. */
        auto st = std::make_shared<HttpResponse>(std::move(res));
        hres.set_chunked_content_provider(
            st->content_type,
            [st](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::string chunk;
                bool more = st->next(chunk);
                if (!chunk.empty()) {
                    if (!sink.write(chunk.data(), chunk.size())) return false;  /* peer gone */
                }
                /* The end is done() and true: false cancels the response, and httplib closes a
                 * connection the client would reuse for its next request. */
                if (!more) sink.done();
                return true;
            },
            [st](bool ok) { if (st->on_close) st->on_close(ok); });
    };

    /* A body is read straight into the string the handler parses. httplib's own Request hands
     * it out only const, so it would be copied and held twice. A multipart body (an image
     * upload) is read part by part, each into its own string, under the same limit. */
    auto read_and_bridge = [bridge](const httplib::Request& q, httplib::Response& s,
                                    const httplib::ContentReader& read) {
        std::string body;
        std::vector<FormPart> form;
        const bool ok = q.is_multipart_form_data()
            ? read([&form](const httplib::FormData& f) {
                       form.push_back(FormPart{ f.name, f.filename, f.content_type, {} });
                       return true;
                   },
                   [&form](const char* p, size_t n) {
                       if (!form.empty()) form.back().data.append(p, n);
                       return true;
                   })
            : read([&body](const char* p, size_t n) { body.append(p, n); return true; });
        if (!ok) {
            /* httplib has already set 413 for a body over the limit; anything else is a peer
             * that stopped sending partway. */
            if (s.status < 0) {
                s.status = 400;
                s.set_content(error_body(q.path, 400, "invalid_request_error",
                                         "the request body could not be read"),
                              "application/json");
            }
            return;
        }
        bridge(q, std::move(body), s, std::move(form));
    };

    /* The preflight allows whatever headers it is asked about. Browser clients send more than
     * the two this server reads (the OpenAI SDK adds x-stainless-*), and an unlisted one fails
     * the request. `*` cannot stand for Authorization in CORS, so the requested list is echoed. */
    auto preflight = [cors](const httplib::Request& req, httplib::Response& res) {
        if (cors) {
            const std::string asked = req.get_header_value("Access-Control-Request-Headers");
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Headers", asked.empty() ? kCorsHeaders : asked);
            res.set_header("Access-Control-Allow-Methods", kCorsMethods);
            res.set_header("Access-Control-Max-Age", "86400");
        }
        res.status = 204;
    };

    /* Socket routes. An upgrade is gated before httplib answers 101 by the same key, carried the
     * ways a socket client can carry it, and, with CORS off, by its Origin: a browser lets any
     * page open a socket to any host, and CORS is what would otherwise keep another site's page
     * out. True when the upgrade may proceed; false with the refusal in `s`. */
    auto socket_gate = [api_key, cors, key_ok](const httplib::Request& q,
                                               httplib::Response& s) -> bool {
        auto refuse = [&](int status, const char* type, const std::string& why) {
            RAD_INFO("req refused from %s:%d -> %d %s: %s", q.remote_addr.c_str(), q.remote_port,
                     status, type, why.c_str());
            s.status = status;
            s.set_content(error_body(q.path, status, type, why), "application/json");
            return false;
        };
        if (!cors && q.has_header("Origin")) {
            std::string origin = q.get_header_value("Origin");
            const size_t at = origin.find("://");
            if (at != std::string::npos) origin = origin.substr(at + 3);
            if (lower_ascii(origin) != lower_ascii(q.get_header_value("Host")))
                return refuse(403, "permission_error",
                              "a page on another origin may not open this socket (--no-cors)");
        }
        if (!api_key.empty() && !key_ok(q) &&
            !offers_key(q.get_header_value("Sec-WebSocket-Protocol"), api_key))
            return refuse(401, "authentication_error", "invalid api key");
        return true;
    };
    for (const std::string& path : router.socket_routes()) {
        const SocketHandler h = *router.socket_for(path);
        svr.WebSocket(
            path,
            [h](const httplib::Request& hreq, httplib::ws::WebSocket& ws) {
                HttpRequest req;
                req.method = hreq.method;
                req.path = hreq.path;
                req.peer = hreq.remote_addr + ":" + std::to_string(hreq.remote_port);
                for (auto& hd : hreq.headers) req.headers[hd.first] = hd.second;
                for (auto& p : hreq.params) req.query[p.first] = p.second;
                HttplibConn conn(ws);
                try {
                    h(req, conn);
                } catch (const std::exception& e) {
                    RAD_ERR("server: %s on %s", e.what(), req.path.c_str());
                    conn.close(1011, "internal error");
                }
            },
            /* `realtime` when it is offered, as OpenAI's server answers; otherwise none. */
            [](const std::vector<std::string>& offered) -> std::string {
                for (const auto& p : offered)
                    if (p == "realtime") return p;
                return std::string();
            });
    }

    /* The router routes every request, not httplib: registered under its own method, a route
     * would make httplib answer the same path under another method 404, where the router answers
     * 405 (http.h). So a request is answered here, before httplib routes it, unless a route takes
     * it with a body to read: httplib hands its body reader only to a route's handler, so those
     * routes are registered below and left to it. A request no route takes is answered without
     * its body being read. */
    svr.set_pre_routing_handler([&router, gate, socket_gate, bridge,
                                 preflight](const httplib::Request& q, httplib::Response& s) {
        /* A WebSocket upgrade to a socket route is left to httplib, which then answers 101 and
         * runs the route's socket handler. */
        if (is_socket_upgrade(q) && router.socket_for(q.path))
            return socket_gate(q, s) ? httplib::Server::HandlerResponse::Unhandled
                                     : httplib::Server::HandlerResponse::Handled;
        if (q.method == "OPTIONS") {
            preflight(q, s);
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!gate(q, s)) return httplib::Server::HandlerResponse::Handled;
        if (q.method != "GET" && q.method != "HEAD" && router.allows(q.method, q.path))
            return httplib::Server::HandlerResponse::Unhandled;
        bridge(q, std::string(), s);
        return httplib::Server::HandlerResponse::Handled;
    });
    for (const auto& sig : router.routes()) {
        const size_t sp = sig.find(' ');
        const std::string method = sig.substr(0, sp), path = sig.substr(sp + 1);
        if (method == "POST") svr.Post(path, read_and_bridge);
        else if (method == "PUT") svr.Put(path, read_and_bridge);
        else if (method == "PATCH") svr.Patch(path, read_and_bridge);
        else if (method == "DELETE") svr.Delete(path, read_and_bridge);
    }

    /* What httplib answers itself (a malformed request, a body over the limit, a body sent to a
     * path its route pattern does not spell) has no body. Give such errors an OpenAI error object
     * like every other refusal; a response that already has a body (the router's 404 and 405,
     * the gate's 401) is left as it is. */
    svr.set_error_handler([cors](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        if (cors) res.set_header("Access-Control-Allow-Origin", "*");
        const std::string type = res.status == 404 ? "not_found_error"
                               : res.status >= 500 ? "internal_server_error"
                                                   : "invalid_request_error";
        const std::string what = res.status == 404 ? "no route for " + req.method + " " + req.path
                                                   : httplib::status_message(res.status);
        res.set_content(error_body(req.path, res.status, type.c_str(), what),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });

    /* Bind once: binding a named port and then any port would leave the first socket bound and
     * unused. A stop that arrived before this point is remembered by the listener and consumed
     * when it starts (httplib's stop_requested_), so nothing here has to look for it. */
    const int port = o.port == 0 ? svr.bind_to_any_port(o.host.c_str())
                     : svr.bind_to_port(o.host.c_str(), o.port) ? o.port : -1;
    if (port <= 0) {
        RAD_ERR("server: cannot bind %s:%d", o.host.c_str(), o.port);
        return RAD_E_IO;
    }
    impl_->port.store(port);
    RAD_INFO("server: listening on http://%s:%d (%d threads, up to %zu)", o.host.c_str(), port,
             nt, most);
    rad_test_before_listen();
    const bool ok = svr.listen_after_bind();
    return ok ? RAD_OK : RAD_E_IO;
}

}  /* namespace server */
}  /* namespace rad */
