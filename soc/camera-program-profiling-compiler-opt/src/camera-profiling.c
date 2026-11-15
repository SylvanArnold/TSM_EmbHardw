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

static void grayscaleFilter(uint32_t width, uint32_t height) {
    for (uint32_t line = 0; line < height; line++) {
#pragma GCC unroll 4 // partial unroll of the pixel loop
        for (uint32_t pixel = 0; pixel < width; pixel++) {
            uint16_t rgb = swap_u16(rgb565[line * width + pixel]); // read encoded rgb pixel

            // extract RGB
            uint32_t red = ((rgb >> 11) & 0x1F) << 3;
            uint32_t green = ((rgb >> 5) & 0x3F) << 2;
            uint32_t blue = (rgb & 0x1F) << 3;

            // Compute grayscale
            uint8_t gray = (uint8_t)((red * 54 + green * 183 + blue * 19) >> 8);
            grayscale[line * width + pixel] = gray;
        }
    }
}

// always_inline: inlined at every call site, like a macro, but type-checked
static inline __attribute__((always_inline)) uint32_t sobelKernel(uint32_t line, uint32_t width, uint32_t pixel) {
    // clang-format off
    const int32_t gx_array[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int32_t gy_array[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };
    // clang-format on

    int32_t valueX = 0;
    int32_t valueY = 0;

#pragma GCC unroll 3 // fully unroll both 3x3 kernel loops
    for (int32_t dx = -1; dx < 2; dx++) {
#pragma GCC unroll 3
        for (int32_t dy = -1; dy < 2; dy++) {
            uint32_t index = ((line + ((uint32_t)dy)) * width) + ((uint32_t)dx) + pixel;
            int32_t gray = (int32_t)(grayscale[index]);
            valueX += gray * gx_array[dy + 1][dx + 1];
            valueY += gray * gy_array[dy + 1][dx + 1];
        }
    }

    int32_t result = (valueX < 0) ? -valueX : valueX;
    result += (valueY < 0) ? -valueY : valueY;

    return (uint32_t)result;
}

static void sobelFilter(uint32_t width, uint32_t height, uint32_t threshold) {
    for (uint32_t line = 1; line < height - 1; line++) {
#pragma GCC unroll 4 // partial unroll of the pixel loop
        for (uint32_t pixel = 1; pixel < width - 1; pixel++) {
            uint32_t result = sobelKernel(line, width, pixel);
            sobel[line * width + pixel] = (result > threshold) ? 0xFF : 0;
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
    grayscaleFilter(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Grayscale cycles");
    perf_print_time(PERF_COUNTER_RUNTIME, "Grayscale time");

    // measure the complete sobel filter
    perf_start();
    sobelFilter(camParams.nrOfPixelsPerLine, camParams.nrOfLinesPerImage, 128);
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
    kernel_result = sobelKernel(test_line, camParams.nrOfPixelsPerLine, test_pixel);
    perf_stop();
    delay_blocking_usec(1);
    perf_print_cycles(PERF_COUNTER_RUNTIME, "Single sobel kernel cycles");
    printf("Single sobel kernel result     : %lu\n", (unsigned long)kernel_result);

    return 0;
}

