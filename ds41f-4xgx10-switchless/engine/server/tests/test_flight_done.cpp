// The flight's DONE, once, on both paths (test_flight_done.py): the single-flight path (decode_one, where
// run() calls finish() alone) and the batch path (decode_batch, where the loop's OnStop calls finish_early()
// inside the run and finish() runs at its end). finish_early and finish's head are the real ones.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace fe {
enum : uint32_t { kStop, kLength, kCancelled, kContext, kError };
}  // namespace fe

static int g_fail = 0;
#define CHECK(c)                                                   \
  do {                                                             \
    if (!(c)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++g_fail;                                                    \
    }                                                              \
  } while (0)

// what done() was told, in order
struct Sent {
  std::vector<uint32_t> reason;
  std::vector<uint32_t> sent;
};

struct Request {
  uint64_t req_id = 0;
};

struct Flight {
  bool ended = false;
  double t_end = 0;
  uint32_t why_sent = 0;
  uint32_t q = 0;
  std::vector<int32_t> out;
  double t_start = 0, t_first = 0;
  Request rq;
};

struct Fixture {
  Sent sent;
  double clock = 100.0;
  double now_s() { return clock += 0.5; }
  void done(const Request&, uint32_t reason, uint32_t, uint32_t n, double, double) {
    sent.reason.push_back(reason);
    sent.sent.push_back(n);
  }

#include "finish_early.inc"

  // finish()'s head: the DONE, then the reason the state ops use
  uint32_t state_why = 0;  // what the state ops below would use
  void finish(Flight& f, uint32_t why) {
#include "finish_head.inc"
    state_why = why;
  }
};

int main() {
  // the single-flight path: run() calls finish() alone, at the run's end
  for (const uint32_t w : {fe::kStop, fe::kLength, fe::kContext, fe::kCancelled}) {
    Fixture x;
    Flight f;
    f.out = {7, 8, 9};
    x.finish(f, w);
    CHECK(x.sent.reason.size() == 1);
    CHECK(x.sent.reason.size() == 1 && x.sent.reason[0] == w);
    CHECK(x.sent.sent.size() == 1 && x.sent.sent[0] == 3);
    CHECK(f.ended);
    CHECK(f.why_sent == w);
    CHECK(f.t_end > 0);
  }

  // the batch path: the loop's OnStop sends it inside the run, finish() at the run's end sends nothing more
  {
    Fixture x;
    Flight f;
    f.out = {7, 8, 9};
    x.finish_early(f, fe::kStop);
    CHECK(x.sent.reason.size() == 1);
    const double t_end = f.t_end;
    x.clock += 10;  // the run's other flights decode on
    x.finish(f, fe::kStop);
    CHECK(x.sent.reason.size() == 1);  // not twice
    CHECK(f.t_end == t_end);           // the DONE's time, not the run's end
  }

  // the batch path, a reason that disagrees: the one the client was told wins
  {
    Fixture x;
    Flight f;
    f.out = {7, 8, 9};
    x.finish_early(f, fe::kStop);
    x.finish(f, fe::kLength);  // a recompute that disagrees
    CHECK(x.sent.reason.size() == 1 && x.sent.reason[0] == fe::kStop);
    CHECK(f.why_sent == fe::kStop);
    CHECK(x.state_why == fe::kStop);  // the state ops use what the client was told
  }

  // the single-flight path: the reason it was called with is the one both use
  {
    Fixture x;
    Flight f;
    f.out = {7};
    x.finish(f, fe::kLength);
    CHECK(x.sent.reason.size() == 1 && x.sent.reason[0] == fe::kLength);
    CHECK(x.state_why == fe::kLength);
  }

  // finish_early twice is one DONE
  {
    Fixture x;
    Flight f;
    f.out = {7};
    x.finish_early(f, fe::kStop);
    x.finish_early(f, fe::kLength);
    CHECK(x.sent.reason.size() == 1 && x.sent.reason[0] == fe::kStop);
  }

  if (g_fail) {
    std::fprintf(stderr, "test_flight_done: %d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_flight_done: ok\n");
  return 0;
}
