// // pykineto2.cpp  (≈120 LoC)
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <kineto/libkineto.h>
// #include <sstream>
// #include <set>

// namespace py = pybind11;
// using   namespace libkineto;
// // Replace the activity-set construction in configure()

// static const std::set<ActivityType> kSafeDefaults = {
//     ActivityType::CPU_OP,
//     ActivityType::USER_ANNOTATION,
//     ActivityType::GPU_USER_ANNOTATION,
//     ActivityType::GPU_MEMCPY,
//     ActivityType::GPU_MEMSET,
//     ActivityType::CONCURRENT_KERNEL,
//     ActivityType::CUDA_RUNTIME,
//     ActivityType::CUDA_DRIVER,
//     ActivityType::OVERHEAD,
//     ActivityType::CUDA_SYNC,
//     ActivityType::COLLECTIVE_COMM};

// class Profiler {
//  public:
//   Profiler() { api().initProfilerIfRegistered(); }

//   // ------------------------------------------------------------
//   // configure
//   // ------------------------------------------------------------
//   void configure(std::vector<std::string> activities   = {},
//                  std::vector<std::string> metrics      = {},
//                  bool  per_kernel        = false,
//                  int   warmup_sec        = 0,
//                  std::string trace_id    = "") {

//     // ----- Activity set ---------------------------------------
//     activities_.clear();
//     if (activities.empty()) {
//       activities_ = kSafeDefaults;
//     } else {
//       for (auto& s : activities) activities_.insert(toActivityType(s));
//     }

//     // ----- Build the key-value block --------------------------
//     std::ostringstream cfg;
//     cfg << "ACTIVITIES_WARMUP_PERIOD_SECS=" << warmup_sec << '\n';

//     if (!metrics.empty()) {
//       cfg << "CUPTI_PROFILER_METRICS=";
//       for (size_t i = 0; i < metrics.size(); ++i) {
//         if (i) cfg << ',';
//         cfg << metrics[i];
//       }
//       cfg << '\n'
//           << "CUPTI_PROFILER_ENABLE_PER_KERNEL="
//           << (per_kernel ? "true" : "false") << '\n';
//     }
//     if (!trace_id.empty()) {
//       cfg << "REQUEST_TRACE_ID="         << trace_id << '\n'
//           << "REQUEST_GROUP_TRACE_ID="   << trace_id << '\n';
//     }
//     config_ = cfg.str();
//   }

//   // ------------------------------------------------------------
//   void start() {
//     api().activityProfiler().prepareTrace(activities_, config_);
//     api().activityProfiler().startTrace();
//   }
//   void stop_to_file(const std::string& path = "trace.json") {
//     api().activityProfiler().stopTrace()->save(path);
//   }

//  private:
//   std::set<ActivityType> activities_;
//   std::string            config_;
// };

// PYBIND11_MODULE(pykineto, m) {
//   py::class_<Profiler>(m, "Profiler")
//       .def(py::init<>())
//       .def("configure", &Profiler::configure,
//            py::arg("activities")   = std::vector<std::string>{},
//            py::arg("metrics")      = std::vector<std::string>{},
//            py::arg("per_kernel")   = false,
//            py::arg("warmup_sec")   = 0,
//            py::arg("trace_id")     = "")
//       .def("start",        &Profiler::start)
//       .def("stop_to_file", &Profiler::stop_to_file,
//            py::arg("path") = "trace.json");
// }

// safe_pykineto.cpp  (drop-in replacement)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <kineto/libkineto.h>
#include <set>
#include <sstream>
#include <iostream>
#include <cuda_runtime_api.h> // add at the top

extern "C" void libkineto_init(bool cpuOnly, bool logOnError);

namespace py = pybind11;
using namespace libkineto;
using libkineto::ActivityType;
using libkineto::api;

static void warmup()
{
  const size_t bytes = 1000;
  void *mem = nullptr; // option ❷
  cudaError_t err = cudaMalloc(&mem, bytes);
  if (err != cudaSuccess)
  {
    std::fprintf(stderr,
                 "kineto warm-up: cudaMalloc failed (%s)\n",
                 cudaGetErrorString(err));
    return;
  }
  cudaFree(mem);
}

// -------- activity kinds that exist on every consumer NVIDIA driver ------
static const std::set<ActivityType> kCudaCpuTypes = {
    ActivityType::CPU_OP,
    ActivityType::USER_ANNOTATION,
    ActivityType::GPU_USER_ANNOTATION,
    ActivityType::GPU_MEMCPY,
    ActivityType::GPU_MEMSET,
    ActivityType::CONCURRENT_KERNEL,
    ActivityType::CUDA_RUNTIME,
    ActivityType::CUDA_DRIVER};

static const std::set<libkineto::ActivityType> types_cupti_prof = {
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::CUDA_PROFILER_RANGE,
};

class Profiler
{
public:
  Profiler()
  {
    warmup();
    libkineto_init(/*cpuOnly=*/false, /*logOnError=*/false);
    api().initProfilerIfRegistered(); // wires the real profiler
    profiler_ = &api().activityProfiler();
  }

  void configure(std::vector<std::string> activities = {},
                 std::vector<std::string> metrics = {},
                 bool per_kernel = false,
                 int warmup_sec = 0,
                 std::string trace_id = "",
                 int verbose_level = 0)
  {

    // ---- Activity set ---------------------------------------------------
    acts_.clear();
    if (activities.empty())
    {
      acts_ = types_cupti_prof;
    }
    else
    {
      for (auto &s : activities)
        acts_.insert(toActivityType(s));
    }
    std::stringstream configss;

    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n";
    // configss << "OUTPUT_DIR=/tmp/kineto_traces";

    // // ---- key=value block ------------------------------------------------
    // std::ostringstream kv;
    // kv << "ACTIVITIES_WARMUP_PERIOD_SECS=" << warmup_sec << '\n'
    //    << "VERBOSE_LOG_LEVEL=" << verbose_level << '\n';

    if (!metrics.empty())
    {
      configss << "CUPTI_PROFILER_METRICS=";
      for (size_t i = 0; i < metrics.size(); ++i)
      {
        if (i)
          configss << ',';
        configss << metrics[i];
      }
      configss << '\n'
               << "CUPTI_PROFILER_ENABLE_PER_KERNEL=" << (per_kernel ? "true" : "false") << '\n';
    }
    // if (!trace_id.empty()) {
    //   kv << "REQUEST_TRACE_ID=" << trace_id << '\n'
    //      << "REQUEST_GROUP_TRACE_ID=" << trace_id << '\n';
    // }

    cfg_ = configss.str();
    std::cout << "Generated config = " << cfg_ << std::endl;
  }

  void start()
  {
    api().resetKinetoTLS(); // match playground logic
    profiler_->prepareTrace(acts_, cfg_);
    warmup(); // let CUPTI settle
    profiler_->startTrace();
  }
  void stop_to_file(const std::string &path = "trace.json")
  {
    if (auto trace = profiler_->stopTrace())
    {
      std::cout << "Kineto captured " << trace->activities()->size()
                << " activities; saving to " << path << "\n";
      trace->save(path);
    }
    else
    {
      std::cerr << "Kineto: stopTrace() returned nullptr!\n";
    }
  }

private:
  std::set<ActivityType> acts_ = types_cupti_prof;
  std::string cfg_ = "";
  libkineto::ActivityProfilerInterface *profiler_ = nullptr; // non‑owning
};

PYBIND11_MODULE(pykineto, m)
{
  py::class_<Profiler>(m, "Profiler")
      .def(py::init<>())
      .def("configure", &Profiler::configure,
           py::arg("activities") = std::vector<std::string>{},
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("per_kernel") = false,
           py::arg("warmup_sec") = 0,
           py::arg("trace_id") = "",
           py::arg("verbose_level") = 0)
      .def("start", &Profiler::start)
      .def("stop_to_file", &Profiler::stop_to_file,
           py::arg("path") = "trace.json");
}
