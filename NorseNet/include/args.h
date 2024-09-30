/*
    Copyright (c) 2020, RootHarold
    All rights reserved.

    Use of this source code is governed by a LGPL-3.0 license that can be found
    in the LICENSE file.
*/

#ifndef NORSE_ARGS_H
#define NORSE_ARGS_H

#include <string>
#include <vector>
#include <thread>
#include <climits>
#include "utils.h"

namespace NorseNet {

    class NorseUtils;

    /*
     * The class Args stores the super parameters needed in
     * various calculation processes of neural networks.
     */

    class Args {
    public:
        Args();

        ~Args();

        friend class NorseUtils;

        friend class Norse;

        friend class Individual;

        friend Norse *loadModel(const std::string &path, uint32_t capacity);

        friend Norse *loadViaString(const std::string &model, uint32_t capacity);

    private:
        // An object of the class NorseUtils which is integrated into this.
        NorseUtils *utils;



    };

}

#endif //NORSE_ARGS_H
