/*
    Copyright (c) 2020, RootHarold
    All rights reserved.

    Use of this source code is governed by a LGPL-3.0 license that can be found
    in the LICENSE file.
*/

#ifndef LYCORIS_ARGS_H
#define LYCORIS_ARGS_H

#include <string>
#include <vector>
#include <thread>
#include <climits>
#include "utils.h"

namespace LycorisNet {

    class LycorisUtils;

    /*
     * The class Args stores the super parameters needed in
     * various calculation processes of neural networks.
     */

    class Args {
    public:
        Args();

        ~Args();

        friend class LycorisUtils;

        friend class Lycoris;

        friend class Individual;

        friend Lycoris *loadModel(const std::string &path, uint32_t capacity);

        friend Lycoris *loadViaString(const std::string &model, uint32_t capacity);

    private:
        // An object of the class LycorisUtils which is integrated into this.
        LycorisUtils *utils;



    };

}

#endif //LYCORIS_ARGS_H
