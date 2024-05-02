#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

int checkP2(int x) {
    int i;
    for (i = 0; (1 << i) <= x; i++);
    return i - 1;
}

int *dTob(int n, int l) {
    int *b = (int *)malloc(l * sizeof(int));
    int i;
    for (i = l - 1; i >= 0; i--) {
        b[i] = n % 2;
        n /= 2;
    }
    return b;
}

int bTod(int a[], int l) {
    int d = 0, i;
    for (i = 0; i < l; i++) {
        d += a[i] * (1 << (l - i - 1));
    }
    return d;
}

int main() {
    int pgs = 4; // page size
    int ms = 32; // memory size
    int nof = ms / pgs; // number of frames
    int offset_bit; // find out # of bits required for offset
    int m; // find out address spaces required in main memory
    int pg_num_bit; // find out # of bits required for page number

    int la[] = {8, 4, 3, 2, 15, 18, 25}; // logical addresses generated by the CPU
    int pmt[] = {3, 6, 8, 12, 2}; // page table

    offset_bit = checkP2(pgs);
    pg_num_bit = checkP2(nof);
    m = 1+ (pgs);

    printf("Page size: %d\n", pgs);
    printf("Memory size: %d\n", ms);
    printf("Number of frames required: %d\n", nof);
    printf("Page number bits: %d\n", pg_num_bit);
    printf("Offset bits: %d\n", offset_bit);
    printf("Number of address spaces: %d\n", m);

    printf("Page Table_____\n");
    for (int i = 0; i < 5; i++) {
        printf("%d -> %d\n", i, pmt[i]);
    }

    for (int i = 0; i < 7; i++) {
        int pg_num = la[i] >> offset_bit;
        int offset = la[i] - (pg_num << offset_bit);

        if (pg_num > 4) {
            printf("%d is an invalid page number\n", pg_num);
            continue;
        }

        int frame_num = pmt[pg_num];
        int physical_addr = (frame_num << offset_bit) + offset;

        if (physical_addr >= ms) {
            printf("%d is an invalid physical address\n", physical_addr);
        } else {
            printf("Corresponding physical address of logical address %d: %d\n", la[i], physical_addr);
        }
    }

    return 0;
}