#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

struct PPMImage {
    size_t x;
    size_t y;
    uint8_t *data;
};


PPMImage * readPPM(const char *path)
{
    // WARNING - heavily untested and not error checked code
    //  also doesn't check for comment lines
    PPMImage *img = NULL;
    FILE *myfile = fopen(path, "r");

    char magic_number[16];
    int x, y, high;
    int c;
    size_t count;

    fscanf(myfile, "%15s", magic_number);
    if (strncmp(magic_number, "P6", 16) != 0) {
        fprintf(stderr, "readPPM only expects P6\n");
        goto cleanup;
    }

    fscanf(myfile, "%d %d", &x, &y);
    fscanf(myfile, "%d", &high);

    if (high != 255) {
        fprintf(stderr, "readPPM only expects max_val of 255\n");
        goto cleanup;
    }

    // skip whitespace
    c = fgetc(myfile);
    while (!feof(myfile) && isspace(c)) {
        c = fgetc(myfile);
    }
    ungetc(c, myfile);

    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (img == NULL)
        goto cleanup;
    img->x = x;
    img->y = y;
    img->data = (uint8_t *)malloc(sizeof(uint8_t) * x * y);
    count = fread(img->data, sizeof(uint8_t), x * y, myfile);

    if (count != x * y * sizeof(uint8_t)) {
        fprintf(stderr, "readPPM read incorrect number of bytes for data\n");
        free(img->data);
        free(img);
        img = NULL;
    }

cleanup:
    fclose(myfile);
    return img;
}


void freePPM(PPMImage *img)
{
    free(img->data);
    free(img);
}






int main(int argc, char const *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [raw-image]\n");
        exit(1);
    }

    PPMImage * img = readPPM(argv[1]);


    freePPM(img);

    return 0;
}
