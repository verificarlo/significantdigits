#include <stdint.h>
#include <stdlib.h>

#include "float_struct.h"

/* truncate x to n bits */
/* binary16 version */
/* binary16 is simulated */
float truncate_binary16(float x, int n) {
  binary32 x_b32 = {.f32 = x};
  /* HALF_PMAN_SIZE = 10 */
  /* FLOAT_PMAN_SIZE = 23 */
  /* We cut the last 13 bits ~ e000 */
  uint32_t mask = 0xFFFFE000 << (FLOAT_PMAN_SIZE - n);
  x_b32.ieee.mantissa &= mask;
  return x_b32.f32;
}

/* truncate x to n bits */
/* binary32 version */
float truncate_binary32(float x, int n) {
  binary32 x_b32 = {.f32 = x};
  uint32_t mask = 0xFFFFFFFF << (FLOAT_PMAN_SIZE - n);
  x_b32.ieee.mantissa &= mask;
  return x_b32.f32;
}

/* truncate x to n bits */
/* binary64 version */
double truncate_binary64(double x, int n) {
  binary64 x_b64 = {.f64 = x};
  uint64_t mask = 0xFFFFFFFFFFFFFFFF << (DOUBLE_PMAN_SIZE - n);
  x_b64.ieee.mantissa &= mask;
  return x_b64.f64;
}
