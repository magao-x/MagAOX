/** \file windsoccImportProbe.cpp
 * \brief Minimal embedded-Python import probe for WindsoCC realtime dependencies.
 * \author Jay Kueny
 *
 * \ingroup windsoccRT_files
 */

#include <Python.h>

#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

using importStage = std::pair<std::string, std::vector<std::string>>;

volatile sig_atomic_t g_shutdownRequested = 0;

/// Minimal signal handler used to emulate MagAO-X app signal context.
void probeSignalHandler(int signum)
{
    g_shutdownRequested = signum;
}

/// Ordered staged imports that mirror the realtime dependency chain.
std::vector<importStage> stagedImports()
{
    return {
        {"numpy", {"numpy"}},
        {"astropy.io.fits", {"astropy.io.fits"}},
        {"yaml", {"yaml"}},
        {"scipy", {"scipy.ndimage", "scipy.signal"}},
        {"matplotlib", {"matplotlib.pyplot"}},
        {"scikit-image", {"skimage.feature", "skimage.measure"}},
        {"pandas", {"pandas"}},
        {"polars", {"polars"}},
        {"sep", {"sep"}},
        {"windsocc.distill", {"windsocc.distill"}},
        {"windsocc.measure", {"windsocc.measure"}},
        {"windsocc.reduce", {"windsocc.reduce"}},
        {"windsocc.xcorr", {"windsocc.xcorr"}},
        {"windsocc.realtime", {"windsocc.realtime"}},
    };
}

/// Print a brief usage message.
void printUsage(const char *argv0)
{
    std::cerr << "Usage: " << argv0
              << " [--python-import-root PATH] [--module MODULE] [--stepwise]"
              << " [--spawn-thread] [--install-signal-handlers] [--report-ids]" << std::endl;
}

/// Install a simple SIGTERM/SIGQUIT/SIGINT handler set similar to MagAOXApp.
int installSignalHandlers()
{
    struct sigaction act;
    sigset_t set;

    act.sa_sigaction = nullptr;
    act.sa_handler = probeSignalHandler;
    act.sa_flags = 0;
    sigemptyset(&set);
    act.sa_mask = set;

    if(sigaction(SIGTERM, &act, nullptr) < 0)
    {
        std::perror("windsoccImportProbe: sigaction(SIGTERM)");
        return -1;
    }

    if(sigaction(SIGQUIT, &act, nullptr) < 0)
    {
        std::perror("windsoccImportProbe: sigaction(SIGQUIT)");
        return -1;
    }

    if(sigaction(SIGINT, &act, nullptr) < 0)
    {
        std::perror("windsoccImportProbe: sigaction(SIGINT)");
        return -1;
    }

    std::cerr << "windsoccImportProbe: installed SIGTERM/SIGQUIT/SIGINT handlers" << std::endl;
    return 0;
}

/// Print real/effective user and group ids for context comparison.
void reportIds()
{
    std::cerr << "windsoccImportProbe: uid=" << getuid() << " euid=" << geteuid()
              << " gid=" << getgid() << " egid=" << getegid() << std::endl;

#ifdef __linux__
    uid_t ruid{0}, euid{0}, suid{0};
    gid_t rgid{0}, egid{0}, sgid{0};

    if(getresuid(&ruid, &euid, &suid) == 0)
    {
        std::cerr << "windsoccImportProbe: ruid/euid/suid=" << ruid << '/' << euid << '/' << suid << std::endl;
    }

    if(getresgid(&rgid, &egid, &sgid) == 0)
    {
        std::cerr << "windsoccImportProbe: rgid/egid/sgid=" << rgid << '/' << egid << '/' << sgid << std::endl;
    }
#endif
}

/// Prepend a root path to Python's import path.
int prependImportRoot(const std::string &pythonImportRoot)
{
    if(pythonImportRoot.empty())
    {
        return 0;
    }

    PyObject *sysPath = PySys_GetObject("path");
    if(sysPath == nullptr)
    {
        PyErr_Print();
        std::cerr << "windsoccImportProbe: failed to access sys.path" << std::endl;
        return -1;
    }

    PyObject *importRoot = PyUnicode_FromString(pythonImportRoot.c_str());
    if(importRoot == nullptr)
    {
        PyErr_Print();
        std::cerr << "windsoccImportProbe: failed to create import-root string" << std::endl;
        return -1;
    }

    if(PySequence_Contains(sysPath, importRoot) == 0)
    {
        if(PyList_Insert(sysPath, 0, importRoot) != 0)
        {
            Py_DECREF(importRoot);
            PyErr_Print();
            std::cerr << "windsoccImportProbe: failed to prepend import root " << pythonImportRoot << std::endl;
            return -1;
        }
    }

    Py_DECREF(importRoot);
    return 0;
}

/// Import a single module and report success/failure.
int importModule(const std::string &moduleName)
{
    std::cerr << "windsoccImportProbe: importing " << moduleName << std::endl;

    PyObject *module = PyImport_ImportModule(moduleName.c_str());
    if(module == nullptr)
    {
        PyErr_Print();
        std::cerr << "windsoccImportProbe: import failed for " << moduleName << std::endl;
        return -1;
    }

    Py_DECREF(module);
    std::cerr << "windsoccImportProbe: import ok for " << moduleName << std::endl;
    return 0;
}

/// Run the staged dependency import sequence.
int importStepwise()
{
    auto stages = stagedImports();

    for(size_t n = 0; n < stages.size(); ++n)
    {
        std::cerr << "windsoccImportProbe: stage " << (n + 1) << " " << stages[n].first << std::endl;
        for(const auto &moduleName : stages[n].second)
        {
            if(importModule(moduleName) < 0)
            {
                return -1;
            }
        }
    }

    return 0;
}

} // namespace

/// Entry point for the embedded-Python import probe.
int main(int argc, char **argv)
{
    std::string pythonImportRoot;
    std::string moduleName{"windsocc.realtime"};
    bool stepwise{false};
    bool spawnThread{false};
    bool installSignals{false};
    bool reportIdentity{false};
    std::atomic<bool> stopHelperThread{false};
    std::thread helperThread;

    for(int n = 1; n < argc; ++n)
    {
        std::string arg{argv[n]};
        if(arg == "--python-import-root" && n + 1 < argc)
        {
            pythonImportRoot = argv[++n];
        }
        else if(arg == "--module" && n + 1 < argc)
        {
            moduleName = argv[++n];
        }
        else if(arg == "--stepwise")
        {
            stepwise = true;
        }
        else if(arg == "--spawn-thread")
        {
            spawnThread = true;
        }
        else if(arg == "--install-signal-handlers")
        {
            installSignals = true;
        }
        else if(arg == "--report-ids")
        {
            reportIdentity = true;
        }
        else if(arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "windsoccImportProbe: unrecognized argument " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if(reportIdentity)
    {
        reportIds();
    }

    if(installSignals)
    {
        if(installSignalHandlers() < 0)
        {
            return 1;
        }
    }

    if(spawnThread)
    {
        std::cerr << "windsoccImportProbe: starting helper thread before Py_Initialize" << std::endl;
        helperThread = std::thread([&stopHelperThread]()
                                   {
                                       while(!stopHelperThread.load(std::memory_order_acquire) &&
                                             g_shutdownRequested == 0)
                                       {
                                           std::this_thread::sleep_for(std::chrono::milliseconds(100));
                                       }
                                   });
    }

    std::cerr << "windsoccImportProbe: calling Py_Initialize" << std::endl;
    Py_Initialize();
    if(!Py_IsInitialized())
    {
        stopHelperThread.store(true, std::memory_order_release);
        if(helperThread.joinable())
        {
            helperThread.join();
        }
        std::cerr << "windsoccImportProbe: Py_Initialize failed" << std::endl;
        return 1;
    }

    std::cerr << "windsoccImportProbe: Python version " << Py_GetVersion() << std::endl;
    if(prependImportRoot(pythonImportRoot) < 0)
    {
        stopHelperThread.store(true, std::memory_order_release);
        if(helperThread.joinable())
        {
            helperThread.join();
        }
        Py_Finalize();
        return 1;
    }

    if(!pythonImportRoot.empty())
    {
        std::cerr << "windsoccImportProbe: prepended sys.path with " << pythonImportRoot << std::endl;
    }

    const int result = stepwise ? importStepwise() : importModule(moduleName);

    std::cerr << "windsoccImportProbe: calling Py_Finalize" << std::endl;
    Py_Finalize();

    stopHelperThread.store(true, std::memory_order_release);
    if(helperThread.joinable())
    {
        std::cerr << "windsoccImportProbe: joining helper thread" << std::endl;
        helperThread.join();
    }

    return result == 0 ? 0 : 1;
}
