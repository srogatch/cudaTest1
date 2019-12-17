#include "cuda_interop.h"
#include <cstdio>

int main() {
    printf("Pointer size: %d\n",int(sizeof(void*)));
    TestOnCuda();
    return 0;
}
