

#include "util.h"

#include "printf.h"

/**
 * @brief Dumps avlues as hex and asxii
 * @details 
 * 
 * @param data 
 * @param size 
 */
void utilDumpHex(const void* data, size_t size)
{
  char ascii[17];
  size_t i, j;
  ascii[16] = '\0';
  for (i = 0; i < size; ++i) {
    printf("%02X ", ((unsigned char*)data)[i]);
    if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
      ascii[i % 16] = ((unsigned char*)data)[i];
    } else {
      ascii[i % 16] = '.';
    }
    if ((i+1) % 8 == 0 || i+1 == size) {
      printf(" ");
      if ((i+1) % 16 == 0) {
        printf("|  %s \n", ascii);
      } else if (i+1 == size) {
        ascii[(i+1) % 16] = '\0';
        if ((i+1) % 16 <= 8) {
          printf(" ");
        }
        for (j = (i+1) % 16; j < 16; ++j) {
          printf("   ");
        }
        printf("|  %s \n", ascii);
      }
    }
  }
}

/**
 * @brief Own memcopy implementation
 * @details 
 * 
 * @param dst destination pointer
 * @param src source pointer
 * @param size number of elements to copy
 */
void utilMemcpy(uint8_t *dst, const uint8_t *src, uint16_t size)
{
  while( size-- )
  {
    *dst++ = *src++;
  }
}


// void* __real_malloc(size_t bytes);
// void __real_free(void *ptr);

// #define MALLOC_TRACK_DEPTH_SIZE (16)
// struct _malloc_track {
//     void *aptr[MALLOC_TRACK_DEPTH_SIZE];
//     size_t a[MALLOC_TRACK_DEPTH_SIZE];
//     int aidx;
//     uint32_t n_a;
//     void *fptr[MALLOC_TRACK_DEPTH_SIZE];
//     int fidx;
//     uint32_t n_f;
//     int n_af;
// } malloc_track;

// void* __wrap_malloc(size_t bytes)
// {
//     uint8_t *ptr;
//     ptr = (uint8_t*)__real_malloc(bytes);

//     if (ptr) {
//         malloc_track.n_af++;
//         malloc_track.n_a++;
//         malloc_track.aptr[malloc_track.aidx] = ptr;
//         malloc_track.a[malloc_track.aidx] = bytes;
//         malloc_track.aidx++;

//         if (malloc_track.aidx >= MALLOC_TRACK_DEPTH_SIZE)
//             malloc_track.aidx = 0;
//     }

//     return ptr;
// }

// void __wrap_free(void *ptr)
// {
//     if (ptr) {
//         malloc_track.n_af--;
//         malloc_track.fptr[malloc_track.fidx] = ptr;
//         malloc_track.fidx++;
//         malloc_track.n_f++;
//         if (malloc_track.fidx >= MALLOC_TRACK_DEPTH_SIZE)
//             malloc_track.fidx = 0;
//     }

//     __real_free(ptr);
// }