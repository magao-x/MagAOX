/** \file windsoccImportProbe.cpp
 * \brief Minimal embedded-Python import probe for WindsoCC realtime dependencies.
 * \author Jay Kueny
 *
 * \ingroup windsoccRT_files
 */

#include <Python.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace
{

using importStage = std::pair<std::string, std::vector<std::string>>;

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
              << " [--python-import-root PATH] [--module MODULE] [--stepwise]" << std::endl;
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

    std::cerr << "windsoccImportProbe: calling Py_Initialize" << std::endl;
    Py_Initialize();
    if(!Py_IsInitialized())
    {
        std::cerr << "windsoccImportProbe: Py_Initialize failed" << std::endl;
        return 1;
    }

    std::cerr << "windsoccImportProbe: Python version " << Py_GetVersion() << std::endl;
    if(prependImportRoot(pythonImportRoot) < 0)
    {
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

    return result == 0 ? 0 : 1;
}
