#include <stdio.h>
#include <stdlib.h>
int main(void) { const char *p=getenv("DYLD_PRINT_LIBRARIES"); printf("DYLD_PRINT_LIBRARIES=%s\n", p?p:"<absent>"); return 0; }
