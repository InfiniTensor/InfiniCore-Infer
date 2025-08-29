#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <infinirt.h>

struct ModelWeights;

__C __export void
loadModelWeight(struct ModelWeights *weights, const char *name, void *data);

__C __export void
loadModelWeightDistributed(struct ModelWeights *weights, const char *name, void *data, int *ranks, int nrank);

#endif // WEIGHTS_LOADER_H
