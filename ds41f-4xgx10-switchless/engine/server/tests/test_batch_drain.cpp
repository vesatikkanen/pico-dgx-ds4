// The decode loops' stop decisions (test_batch_drain.py): the batch drain's OnStop contract, on the real
// lambdas of cmd.cpp's batch_loop and spec_batch_loop, with a synthetic entry boundary.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

enum : uint32_t { kEndStop, kEndLength, kEndCancel, kEndContext, kEndPause, kEndSwitch };
using OnStop = std::function<void(int k, uint32_t why)>;

static int g_fail = 0;
#define CHECK(c)                                                   \
  do {                                                             \
    if (!(c)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++g_fail;                                                    \
    }                                                              \
  } while (0)

// what the caller saw, in order
struct Seen {
  std::vector<int> k;
  std::vector<uint32_t> why;
  void operator()(int e, uint32_t w) { k.push_back(e), why.push_back(w); }
};

// ---- batch_loop's stop, verbatim from cmd.cpp ------------------------------------------------
struct BatchFixture {
  int n = 2;
  std::vector<int> launched = {3, 3};
  std::vector<char> live = std::vector<char>(n, 1), ending = std::vector<char>(n, 0);
  OnStop on_stop;
  struct End {
    uint32_t why = kEndStop;
    int fed = 0;
    int sampled = 3;  // the sample it ended after; == launched means nothing is queued
  };
  std::vector<End> end = std::vector<End>(n);

  void run(int k, uint32_t why) {
    auto stop = [&](int k, uint32_t why) {
#include "batch_stop.inc"
    };
    stop(k, why);
  }
};

// ---- spec_batch_loop's finish, verbatim from cmd.cpp -----------------------------------------
struct SpecFixture {
  int n = 2;
  std::vector<char> live = std::vector<char>(n, 1), ending = std::vector<char>(n, 0), pend = std::vector<char>(n, 0);
  struct Entry {
    int pos0 = 0;
  };
  std::vector<Entry> e = std::vector<Entry>(n);
  std::vector<std::vector<int32_t>> toks = {{7, 8, 9}, {7, 8, 9}};
  OnStop on_stop;
  struct End {
    uint32_t why = kEndStop;
    int fed = 0;
    int32_t last = 0;
  };
  std::vector<End> end = std::vector<End>(n);

  void run(int k, int kv_end, uint32_t why) {
    end[k].why = why;  // the loop sets it where it decides the end
    auto finish = [&](int k, int kv_end) {
#include "spec_finish.inc"
    };
    finish(k, kv_end);
  }
};

int main() {
  // batch_loop: a real end is reported once, with the loop's own reason
  for (const uint32_t w : {kEndStop, kEndLength, kEndContext, kEndCancel}) {
    BatchFixture f;
    Seen s;
    f.on_stop = std::ref(s);
    f.run(1, w);
    CHECK(s.k.size() == 1);
    CHECK(s.k.size() == 1 && s.k[0] == 1);
    CHECK(s.why.size() == 1 && s.why[0] == w);
    CHECK(f.end[1].why == w);
    CHECK(f.live[1] == 0);  // nothing queued: it ends now
    CHECK(f.end[1].fed == 3);
  }
  // batch_loop: a pause is not an end
  {
    BatchFixture f;
    Seen s;
    f.on_stop = std::ref(s);
    f.run(0, kEndPause);
    CHECK(s.k.empty());
    CHECK(f.end[0].why == kEndPause);
  }
  // batch_loop: a row still queued keeps it live, but the end is still reported
  {
    BatchFixture f;
    Seen s;
    f.on_stop = std::ref(s);
    f.launched[0] = 5, f.end[0].sampled = 2;
    f.run(0, kEndStop);
    CHECK(s.k.size() == 1 && s.why[0] == kEndStop);
    CHECK(f.ending[0] == 1);
    CHECK(f.live[0] == 1);
  }
  // batch_loop: no OnStop (main.cpp's loops) is not a crash
  {
    BatchFixture f;
    f.run(0, kEndStop);
    CHECK(f.end[0].why == kEndStop);
    CHECK(f.live[0] == 0);
  }

  // spec_batch_loop: a real end is reported once, with the loop's own reason
  for (const uint32_t w : {kEndStop, kEndLength, kEndContext, kEndCancel}) {
    SpecFixture f;
    Seen s;
    f.on_stop = std::ref(s);
    f.run(1, 3, w);
    CHECK(s.k.size() == 1);
    CHECK(s.k.size() == 1 && s.k[0] == 1);
    CHECK(s.why.size() == 1 && s.why[0] == w);
    CHECK(f.end[1].fed == 3);
    CHECK(f.end[1].last == 9);
    CHECK(f.live[1] == 0 && f.ending[1] == 0 && f.pend[1] == 0);
  }
  // spec_batch_loop: a pause and a copy switch are not ends
  for (const uint32_t w : {kEndPause, kEndSwitch}) {
    SpecFixture f;
    Seen s;
    f.on_stop = std::ref(s);
    f.run(0, 3, w);
    CHECK(s.k.empty());
    CHECK(f.end[0].why == w);
    CHECK(f.live[0] == 0);  // it still ends; only the DONE is withheld
  }
  // spec_batch_loop: no OnStop is not a crash
  {
    SpecFixture f;
    f.run(0, 2, kEndStop);
    CHECK(f.end[0].fed == 2);
    CHECK(f.end[0].last == 8);
    CHECK(f.live[0] == 0);
  }

  if (g_fail) {
    std::fprintf(stderr, "test_batch_drain: %d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_batch_drain: ok\n");
  return 0;
}
