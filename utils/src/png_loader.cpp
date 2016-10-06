/*
Copyright 2016 Tommi M. Tykkälä

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#define PNG_SKIP_SETJMP_CHECK
#include <png.h>
#include <assert.h>
#include <stdlib.h>
#include <float.h>

#define PNG_SIG_BYTES 8

unsigned char *loadPNG(const char *name, unsigned int *width, unsigned int *height, unsigned int *nChannels, unsigned int *pitch, bool flipY)
{
    printf("loading image %s\n",name);
	FILE *png_file = fopen(name, "rb");
	assert(png_file);

	unsigned char header[PNG_SIG_BYTES];

	int ret = fread(&header[0], 1, PNG_SIG_BYTES, png_file);
	if (!png_sig_cmp((png_bytep)&header[0], 0, PNG_SIG_BYTES) == 0) {
		printf("png file %s did not exist or not readable!\n",name);
		assert(0);
	}
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(png_ptr);

	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);

	png_infop end_info = png_create_info_struct(png_ptr);
	assert(end_info);

        //assert(!setjmp(png_jmpbuf(png_ptr)));
	png_init_io(png_ptr, png_file);
	png_set_sig_bytes(png_ptr, PNG_SIG_BYTES);
	png_read_info(png_ptr, info_ptr);

	png_uint_32 bit_depth, color_type;
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);

	if (bit_depth == 16)
			png_set_strip_16(png_ptr);
			
	if(color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	else if(color_type == PNG_COLOR_TYPE_GRAY ||
			color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		{
			png_set_gray_to_rgb(png_ptr);
		}

	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png_ptr);
	else
		png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);

	png_read_update_info(png_ptr, info_ptr);

	*nChannels = png_get_channels(png_ptr, info_ptr);
	*pitch = png_get_rowbytes(png_ptr, info_ptr);
	*width = png_get_image_width(png_ptr, info_ptr);
	*height = png_get_image_height(png_ptr, info_ptr);

	png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
	png_uint_32 numbytes = rowbytes*(*height);
	png_byte* pixels = new png_byte[numbytes];
	png_byte** row_ptrs = (png_byte**)malloc((*height) * sizeof(png_byte*));

    if (!flipY) {
        for (unsigned int i=0; i<(*height); i++)
            row_ptrs[i] = pixels + i*rowbytes;
    } else {
        for (int i=0; i<(*height); i++)
            row_ptrs[i] = pixels + ((*height) - 1 - i)*rowbytes;
    }

	png_read_image(png_ptr, row_ptrs);

	free(row_ptrs);
	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(png_file);

    return (unsigned char *)pixels;
}

/* Attempts to save PNG to file; returns 0 on success, non-zero on error. */
int savePNG(unsigned char *data, int width, int height, int nChannels, int pitch, const char *fileName)
{
	FILE *fp = fopen(fileName, "wb");
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	int y;

	if (fp == NULL) return -1;

	/* Initialize the write struct. */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fclose(fp);
		return -1;
	}

	/* Initialize the info struct. */
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		png_destroy_write_struct(&png_ptr, NULL);
		fclose(fp);
		return -1;
	}

	unsigned int colortype = PNG_COLOR_TYPE_RGBA;
	if (nChannels == 1) colortype = PNG_COLOR_TYPE_GRAY;
	else if (nChannels == 3) colortype = PNG_COLOR_TYPE_RGB;
	else if (nChannels == 4) colortype = PNG_COLOR_TYPE_RGBA;
	else assert(0);

	/* Set Image attributes. */
	png_set_IHDR(png_ptr,
		info_ptr,
		width,
		height,
		8,
		colortype,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);

	png_byte** row_ptrs = (png_byte**)malloc(height * sizeof(png_byte*));

	for (y = 0; y < height; ++y) {
		row_ptrs[y] = data + y*pitch;

	}

	/* Actually write the Image data. */
	png_init_io(png_ptr, fp);
	png_set_rows(png_ptr, info_ptr, row_ptrs);
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
	for (int y = 0; y < height; y++){
		free(row_ptrs[y]);
	}
	free(row_ptrs);

	/* Finish writing. */
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
	return 0;
}

