#include "utils.h"

verbose_runtime_error::verbose_runtime_error(const std::string& arg,
                                             const char* file, int line)
    : std::runtime_error(arg) {

    std::string niceName = getFilename(file);

    std::ostringstream o;
    o << arg << " at " << niceName << ":" << line;
    msg = o.str();
}

string getFilename(string filePath) {

    // stolen from polyscope/utilities.cpp
    size_t startInd = 0;
    for (std::string sep : {"/", "\\"}) {
        size_t pos = filePath.rfind(sep);
        if (pos != std::string::npos) {
            startInd = std::max(startInd, pos + 1);
        }
    }

    return filePath.substr(startInd, filePath.size());
}
