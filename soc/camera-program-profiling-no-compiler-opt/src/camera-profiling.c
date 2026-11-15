#include <ov7670.h>
#include <stdio.h>
#include <swap.h>
#include <switches.h>
#include <vga.h>

#include <delay.h>
#include <perf.h>

#define PIXEL_AMOUNT 640 * 480
volatile uint16_t rgb565[PIXEL_AMOUNT];
volatile uint8_t grayscale[PIXEL_AMOUNT];
volatile uint8_t sobel[PIXEL_AMOUNT];

// Number of pixels processed per iteration of the unrolled loops
#define UNROLL_FACTOR 4

// Convert the pixel at index idx from rgb565 to grayscale.
// A macro (not a function) so the unrolled body really is inlined, even at -O0.
#define GRAYSCALE_PIXEL(idx)                                                   \
    do {                                                                       \
        uint16_t rgb = swap_u16(rgb565[(idx)]); /* read encoded rgb pixel */   \
        uint32_t red = ((rgb >> 11) & 0x1F) << 3;                              \
        uint32_t green = ((rgb >> 5) & 0x3F) << 2;                             \
        uint32_t blue = (rgb & 0x1F) << 3;                                     \
        grayscale[(idx)] = (uint8_t)((red * 54 + green * 183 + blue * 19) >> 8); \
    } while (0)

// Convert the whole rgb565 image to grayscale.
// A macro (not a function) so no call is made, even at -O0.
// Arguments are copied into gf_ locals once, so each is evaluated only once.
#define GRAYSCALE_FILTER(width, height)                                            \
    do {                                                                           \
        uint32_t gf_width = (width);                                               \
        uint32_t gf_height = (height);                                             \
        for (uint32_t gf_line = 0; gf_line < gf_height; gf_line++) {               \
            uint32_t gf_idx = gf_line * gf_width;   /* first pixel of the line */  \
            uint32_t gf_lineEnd = gf_idx + gf_width; /* one past the last pixel */ \
            uint32_t gf_unrolledEnd =                                              \
                gf_idx + (gf_width & ~(uint32_t)(UNROLL_FACTOR - 1));              \
                                                                                   \
            /* pixel loop partially unrolled by UNROLL_FACTOR */                   \
            for (; gf_idx < gf_unrolledEnd; gf_idx += UNROLL_FACTOR) {             \
                GRAYSCALE_PIXEL(gf_idx);                                           \
                GRAYSCALE_PIXEL(gf_idx + 1);                                       \
                GRAYSCALE_PIXEL(gf_idx + 2);                                       \
                GRAYSCALE_PIXEL(gf_idx + 3);                                       \
            }                                                                      \
                                                                                   \
            /* remaining pixels (width not a multiple of UNROLL_FACTOR) */         \
            for (; gf_idx < gf_lineEnd; gf_idx++) {                                \
                GRAYSCALE_PIXEL(gf_idx);                                           \
            }                                                                      \
        }                                                                          \
    } while (0)

// clang-format off
static const int32_t gx_array[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
static const int32_t gy_array[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};
// clang-format on

// Compute the sobel kernel at (line, pixel) and assign |Gx| + |Gy| to result.
// A macro (not a function) so the kernel is inlined at every use, even at -O0.
// Locals are prefixed with sk_ so they cannot capture names from the arguments.
#define SOBEL_KERNEL(line, width, pixel, result)                                      \
    do {                                                                              \
        int32_t sk_valueX = 0;                                                        \
        int32_t sk_valueY = 0;                                                        \
        int32_t sk_gray;                                                              \
                                                                                      \
        /* both loops (dx and dy = -1, 0, +1) unrolled by hand */                     \
        uint32_t sk_top = (((line) - 1) * (width)) + (pixel); /* (dx = 0, dy = -1) */ \
        uint32_t sk_mid = sk_top + (width);                   /* (dx = 0, dy =  0) */ \
        uint32_t sk_bot = sk_mid + (width);                   /* (dx = 0, dy = +1) */ \
                                                                                      \
        /* dx = -1 */                                                                 \
        sk_gray = (int32_t)(grayscale[sk_top - 1]); /* dy = -1 */                     \
        sk_valueX += sk_gray * gx_array[0][0];                                        \
        sk_valueY += sk_gray * gy_array[0][0];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_mid - 1]); /* dy = 0 */                      \
        sk_valueX += sk_gray * gx_array[1][0];                                        \
        sk_valueY += sk_gray * gy_array[1][0];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_bot - 1]); /* dy = +1 */                     \
        sk_valueX += sk_gray * gx_array[2][0];                                        \
        sk_valueY += sk_gray * gy_array[2][0];                                        \
                                                                                      \
        /* dx = 0 */                                                                  \
        sk_gray = (int32_t)(grayscale[sk_top]); /* dy = -1 */                         \
        sk_valueX += sk_gray * gx_array[0][1];                                        \
        sk_valueY += sk_gray * gy_array[0][1];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_mid]); /* dy = 0 */                          \
        sk_valueX += sk_gray * gx_array[1][1];                                        \
        sk_valueY += sk_gray * gy_array[1][1];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_bot]); /* dy = +1 */                         \
        sk_valueX += sk_gray * gx_array[2][1];                                        \
        sk_valueY += sk_gray * gy_array[2][1];                                        \
                                                                                      \
        /* dx = +1 */                                                                 \
        sk_gray = (int32_t)(grayscale[sk_top + 1]); /* dy = -1 */                     \
        sk_valueX += sk_gray * gx_array[0][2];                                        \
        sk_valueY += sk_gray * gy_array[0][2];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_mid + 1]); /* dy = 0 */                      \
        sk_valueX += sk_gray * gx_array[1][2];                                        \
        sk_valueY += sk_gray * gy_array[1][2];                                        \
                                                                                      \
        sk_gray = (int32_t)(grayscale[sk_bot + 1]); /* dy = +1 */                     \
        sk_valueX += sk_gray * gx_array[2][2];                                        \
        sk_valueY += sk_gray * gy_array[2][2];                                        \
                                                                                      \
        (result) = (uint32_t)(((sk_valueX < 0) ? -sk_valueX : sk_valueX) +            \
                              ((sk_valueY < 0) ? -sk_valueY : sk_valueY));            \
    } while (0)

// Apply the kernel at column p of the given line and store the thresholded
// result in out (pointer to the start of the output line).
#define SOBEL_PIXEL(line, width, threshold, out, p)        \
    do {                                                   \
        uint32_t sp_result;                                \
        SOBEL_KERNEL((line), (width), (p), sp_result);     \
        (out)[(p)] = (sp_result > (threshold)) ? 0xFF : 0; \
    } while (0)

// Apply the sobel filter on the whole grayscale image (borders are skipped).
// A macro (not a function) so no call is made, even at -O0.
// Arguments are copied into sf_ locals once, so each is evaluated only once.
#define SOBEL_FILTER(width, height, threshold)                                              \
    do {                                                                                    \
        uint32_t sf_width = (width);                                                        \
        uint32_t sf_height = (height);                                                      \
        uint32_t sf_threshold = (threshold);                                                \
        /* pixels 1 .. width-2 are processed */                                             \
        uint32_t sf_innerWidth = sf_width - 2;                                              \
        uint32_t sf_unrolledEnd = 1 + (sf_innerWidth & ~(uint32_t)(UNROLL_FACTOR - 1));     \
                                                                                            \
        for (uint32_t sf_line = 1; sf_line < sf_height - 1; sf_line++) {                    \
            volatile uint8_t* sf_out = &sobel[sf_line * sf_width]; /* output line start */  \
            uint32_t sf_pixel = 1;                                                          \
                                                                                            \
            /* pixel loop partially unrolled by UNROLL_FACTOR */                            \
            for (; sf_pixel < sf_unrolledEnd; sf_pixel += UNROLL_FACTOR) {                  \
                SOBEL_PIXEL(sf_line, sf_width, sf_threshold, sf_out, sf_pixel);             \
                SOBEL_PIXEL(sf_line, sf_width, sf_threshold, sf_out, sf_pixel + 1);         \
                SOBEL_PIXEL(sf_line, sf_width, sf_threshold, sf_out, sf_pixel + 2);         \
                SOBEL_PIXEL(sf_line, sf_width, sf_threshold, sf_out, sf_pixel + 3);         \
            }                                                                               \
                                                                                            \
            /* remaining pixels (innerWidth not a multiple of UNROLL_FACTOR) */             \
            for (; sf_pixel < sf_width - 1; sf_pixel++) {                                   \
                SOBEL_PIXEL(sf_line, sf_width, sf_threshold, sf_out, sf_pixel);             \
            }                                                                               \
        }                                                                                   \
    } while (0)

int main() {

    printf("Starting camera software profiling program\n");

    volatile uint32_t* switches = (uint32_t*)SWITCHES_BASE_ADDRESS;
    volatile uint32_t* vga = (uint32_t*)VGA_BASE_ADDRESS;

    vga_clear();

    printf("Initialising camera (this takes up to 3 seconds)!\n");

    camParameters camParams = initOv7670(VGA);

    printf("Done!\n");
    printf("NrOfPixels : %d\n", camParams.nrOfPixelsPerLine);
    printf("NrOfLines  : %d\n", camParams.nrOfLinesPerImage);
    printf("PCLK (kHz) : %d\n", camParams.pixelClockInkHz);
    printf("FPS        : %d\n", camParams.framesPerSecond);

    // Display configuration
    vga[0] = swap_u32(camParams.nrOfPixelsPerLine);

    perf_init();

    perf_start();
    // capture the camera image
    takeSingleImageBlocking((uint32_t)&rgb565[0]);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Image Capture cycles");
    perf_print_time(PERF_COUNTER_RUNTIME,"Image Capture time");


    vga[1] = swap_u32(camParams.nrOfLinesPerImage);
    vga[2] = swap_u32(2);
    vga[3] = swap_u32((uint32_t)&sobel[0]);


    // measure the grayscale conversion
    perf_start();
    GRAYSCALE_FILTER(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Grayscale cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Grayscale time");

    // measure the complete sobel filter
    perf_start();
    SOBEL_FILTER(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage, 128);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Sobel filter cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Sobel filter time");

    // pre compute the line/pixel where to apply the kernel
    uint32_t test_pixel = camParams.nrOfPixelsPerLine / 2;
    uint32_t test_line = camParams.nrOfLinesPerImage / 2;

    // measure the cycles to compute a single kernel
    // (result stored in a volatile so the compiler cannot drop the computation)
    volatile uint32_t kernel_result;
    perf_start();
    SOBEL_KERNEL(test_line, camParams.nrOfPixelsPerLine, test_pixel, kernel_result);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Single sobel kernel cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Sobel kernel time");

    return 0;
}

