#include <ov7670.h>
#include <stdio.h>
#include <swap.h>
#include <switches.h>
#include <vga.h>

#include <delay.h>
#include <perf.h>

#define PIXEL_AMOUNT (640 * 480)

// Buffers are NOT volatile anymore: volatile forced every access to go to memory
// and prevented the compiler from keeping pixels in registers. The camera DMA
// writes rgb565 inside takeSingleImageBlocking() (external call + explicit
// compiler barrier below), and the VGA only reads sobel, so volatile is not needed.
// aligned(4) allows 32-bit loads/stores on the buffers.
uint16_t rgb565[PIXEL_AMOUNT] __attribute__((aligned(4)));
uint8_t grayscale[PIXEL_AMOUNT] __attribute__((aligned(4)));
uint8_t sobel[PIXEL_AMOUNT] __attribute__((aligned(4)));

#define COMPILER_BARRIER() asm volatile("" ::: "memory")

// perf_start/perf_stop are asm volatile without a "memory" clobber: now that the
// buffers are no longer volatile, the compiler could move loads/stores of the
// filters across them. The barriers keep the measured work inside the window.
#define PERF_START()        \
    do {                    \
        COMPILER_BARRIER(); \
        perf_start();       \
        COMPILER_BARRIER(); \
    } while (0)
#define PERF_STOP()         \
    do {                    \
        COMPILER_BARRIER(); \
        perf_stop();        \
        COMPILER_BARRIER(); \
    } while (0)

// gray = (R8*54 + G8*183 + B8*19) >> 8 with R8 = R5<<3, G8 = G6<<2, B8 = B5<<3.
// The <<3 / <<2 are folded into the coefficients: 54*8 = 432, 183*4 = 732, 19*8 = 152.
// (bit-exact with the previous version)
static inline __attribute__((always_inline)) uint32_t rgb565ToGray(uint32_t rgb) {
    uint32_t red = (rgb >> 11) & 0x1F;
    uint32_t green = (rgb >> 5) & 0x3F;
    uint32_t blue = rgb & 0x1F;
    return (red * 432 + green * 732 + blue * 152) >> 8;
}

// Processes 4 pixels per iteration: 2x 32-bit loads (2 rgb565 pixels each) and
// 1x 32-bit store (4 gray bytes) instead of 4 half-word loads + 4 byte stores.
static void grayscaleFilter(uint32_t width, uint32_t height) {
    const uint32_t* restrict src = (const uint32_t*)rgb565;
    uint32_t* restrict dst = (uint32_t*)grayscale;
    uint32_t nrOfPixels = width * height;

    // OR1K is big-endian: bytes b0 b1 b2 b3 are read as 0xb0b1b2b3.
    // swap_u32 gives 0xb3b2b1b0 -> pixel0 (bytes b1b0) in the low half,
    // pixel1 (bytes b3b2) in the high half: one custom instruction swaps 2 pixels.
    for (uint32_t i = 0; i < nrOfPixels / 4; i++) {
        uint32_t w0 = swap_u32(src[2 * i]);
        uint32_t w1 = swap_u32(src[2 * i + 1]);

        uint32_t g0 = rgb565ToGray(w0 & 0xFFFF);
        uint32_t g1 = rgb565ToGray(w0 >> 16);
        uint32_t g2 = rgb565ToGray(w1 & 0xFFFF);
        uint32_t g3 = rgb565ToGray(w1 >> 16);

        // big-endian: first pixel goes into the most significant byte
        dst[i] = (g0 << 24) | (g1 << 16) | (g2 << 8) | g3;
    }

    // remaining pixels if the image size is not a multiple of 4
    for (uint32_t i = nrOfPixels & ~3u; i < nrOfPixels; i++) {
        grayscale[i] = (uint8_t)rgb565ToGray(swap_u16(rgb565[i]));
    }
}

// Separable Sobel:
//   Gx = S(x+1) - S(x-1)               with S(c) = top[c] + 2*mid[c] + bot[c]
//   Gy = D(x-1) + 2*D(x) + D(x+1)      with D(c) = top[c] - bot[c]
// Only 6 additions instead of 18 multiply-accumulates.
static inline __attribute__((always_inline)) uint32_t sobelKernel(uint32_t line, uint32_t width, uint32_t pixel) {
    const uint8_t* top = &grayscale[(line - 1) * width + pixel];
    const uint8_t* mid = top + width;
    const uint8_t* bot = mid + width;

    int32_t s0 = top[-1] + 2 * mid[-1] + bot[-1];
    int32_t s2 = top[1] + 2 * mid[1] + bot[1];
    int32_t d0 = top[-1] - bot[-1];
    int32_t d1 = top[0] - bot[0];
    int32_t d2 = top[1] - bot[1];

    int32_t gx = s2 - s0;
    int32_t gy = d0 + 2 * d1 + d2;
    return (uint32_t)(((gx < 0) ? -gx : gx) + ((gy < 0) ? -gy : gy));
}

// Sliding window: S/D of columns x-1 and x are kept in registers, so each
// output pixel only loads one new column (3 bytes) instead of 9 bytes.
// Evaluates to 0xFF / 0x00 for the pixel at column x.
#define SOBEL_STEP(x)                                                  \
    ({                                                                 \
        uint32_t t_ = top[(x) + 1], m_ = mid[(x) + 1], b_ = bot[(x) + 1]; \
        int32_t s2_ = (int32_t)(t_ + 2 * m_ + b_);                      \
        int32_t d2_ = (int32_t)t_ - (int32_t)b_;                        \
        int32_t gx_ = s2_ - s0;                                        \
        int32_t gy_ = d0 + 2 * d1 + d2_;                               \
        s0 = s1;                                                       \
        s1 = s2_;                                                      \
        d0 = d1;                                                       \
        d1 = d2_;                                                      \
        uint32_t r_ = (uint32_t)(((gx_ < 0) ? -gx_ : gx_) + ((gy_ < 0) ? -gy_ : gy_)); \
        (r_ > threshold) ? 0xFFu : 0u;                                 \
    })

static void sobelFilter(uint32_t width, uint32_t height, uint32_t threshold) {
    for (uint32_t line = 1; line < height - 1; line++) {
        const uint8_t* restrict top = &grayscale[(line - 1) * width];
        const uint8_t* restrict mid = top + width;
        const uint8_t* restrict bot = mid + width;
        uint8_t* restrict out = &sobel[line * width];

        // prime the window with columns 0 and 1
        int32_t s0 = top[0] + 2 * mid[0] + bot[0];
        int32_t d0 = top[0] - bot[0];
        int32_t s1 = top[1] + 2 * mid[1] + bot[1];
        int32_t d1 = top[1] - bot[1];

        if ((width & 3) == 0 && width >= 8) {
            // Fast path: 32-bit stores of 4 output pixels (big-endian packing).
            // Border pixels (column 0 and width-1) are written as 0.
            uint32_t* restrict out32 = (uint32_t*)out;

            uint32_t w = SOBEL_STEP(1) << 16;
            w |= SOBEL_STEP(2) << 8;
            w |= SOBEL_STEP(3);
            out32[0] = w;

            uint32_t x;
            for (x = 4; x < width - 4; x += 4) {
                w = SOBEL_STEP(x) << 24;
                w |= SOBEL_STEP(x + 1) << 16;
                w |= SOBEL_STEP(x + 2) << 8;
                w |= SOBEL_STEP(x + 3);
                out32[x / 4] = w;
            }

            w = SOBEL_STEP(x) << 24;
            w |= SOBEL_STEP(x + 1) << 16;
            w |= SOBEL_STEP(x + 2) << 8;
            out32[x / 4] = w;
        } else {
            // Generic path: byte stores
#pragma GCC unroll 4
            for (uint32_t x = 1; x < width - 1; x++) {
                out[x] = (uint8_t)SOBEL_STEP(x);
            }
        }
    }
}

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

    PERF_START();
    // capture the camera image
    takeSingleImageBlocking((uint32_t)&rgb565[0]);
    PERF_STOP(); // also acts as barrier: rgb565 was written by the camera DMA
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Image Capture cycles");
    perf_print_time(PERF_COUNTER_RUNTIME,"Image Capture time");


    vga[1] = swap_u32(camParams.nrOfLinesPerImage);
    vga[2] = swap_u32(2);
    vga[3] = swap_u32((uint32_t)&sobel[0]);


    // measure the grayscale conversion
    PERF_START();
    grayscaleFilter(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage);
    PERF_STOP();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Grayscale cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Grayscale time");

    // measure the complete sobel filter
    PERF_START();
    sobelFilter(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage, 128);
    PERF_STOP();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Sobel filter cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Sobel filter time");

    // pre compute the line/pixel where to apply the kernel
    uint32_t test_pixel = camParams.nrOfPixelsPerLine / 2;
    uint32_t test_line = camParams.nrOfLinesPerImage / 2;

    // measure the cycles to compute a single kernel
    // (result stored in a volatile so the compiler cannot drop the computation)
    volatile uint32_t kernel_result;
    PERF_START();
    kernel_result = sobelKernel(test_line, camParams.nrOfPixelsPerLine, test_pixel);
    PERF_STOP();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Single sobel kernel cycles");
    printf("Single sobel kernel result     : %lu\n", (unsigned long)kernel_result);

    return 0;
}
