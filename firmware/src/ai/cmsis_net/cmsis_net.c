#include <stdint.h>
#include "main.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "cmsis_net.h"

// Layer conv1
static const q7_t conv1_wt [CONV1_INPUT_CH*CONV1_KERNEL_X*CONV1_KERNEL_Y*CONV1_OUTPUT_CH] = CONV1_WT;
static const int32_t conv1_bias [CONV1_OUTPUT_CH] = CONV1_BIAS;
static const int32_t conv1_output_mult [CONV1_OUTPUT_CH] = CONV1_OUT_MULT;
static const int32_t conv1_output_shift [CONV1_OUTPUT_CH] = CONV1_OUT_SHIFT;
// Layer conv2
static const q7_t conv2_wt [CONV2_INPUT_CH*CONV2_KERNEL_X*CONV2_KERNEL_Y*CONV2_OUTPUT_CH] = CONV2_WT;
static const int32_t conv2_bias [CONV2_OUTPUT_CH] = CONV2_BIAS;
static const int32_t conv2_output_mult [CONV2_OUTPUT_CH] = CONV2_OUT_MULT;
static const int32_t conv2_output_shift [CONV2_OUTPUT_CH] = CONV2_OUT_SHIFT;
// Layer conv3
static const q7_t conv3_wt [CONV3_INPUT_CH*CONV3_KERNEL_X*CONV3_KERNEL_Y*CONV3_OUTPUT_CH] = CONV3_WT;
static const int32_t conv3_bias [CONV3_OUTPUT_CH] = CONV3_BIAS;
static const int32_t conv3_output_mult [CONV3_OUTPUT_CH] = CONV3_OUT_MULT;
static const int32_t conv3_output_shift [CONV3_OUTPUT_CH] = CONV3_OUT_SHIFT;
// Layer conv4
static const q7_t conv4_wt [CONV4_INPUT_CH*CONV4_KERNEL_X*CONV4_KERNEL_Y*CONV4_OUTPUT_CH] = CONV4_WT;
static const int32_t conv4_bias [CONV4_OUTPUT_CH] = CONV4_BIAS;
static const int32_t conv4_output_mult [CONV4_OUTPUT_CH] = CONV4_OUT_MULT;
static const int32_t conv4_output_shift [CONV4_OUTPUT_CH] = CONV4_OUT_SHIFT;
// Layer fc1
static const q7_t fc1_wt [ sizeof(q15_t) * FC1_COL_DIM * FC1_ROW_DIM] = FC1_WT;
static const int32_t fc1_bias [ sizeof(q15_t) * FC1_ROW_DIM] = FC1_BIAS;

// calculation and activations buffer
static uint8_t tmpBuf[2304];
static uint8_t activations1[3888];
static uint8_t activations2[3888];

arm_status convolve_s8(const q7_t *input,
               const uint16_t input_x,
               const uint16_t input_y,
               const uint16_t input_ch,
               const uint16_t input_batches,
               const q7_t *kernel,
               const uint16_t output_ch,
               const uint16_t kernel_x,
               const uint16_t kernel_y,
               const uint16_t pad_x,
               const uint16_t pad_y,
               const uint16_t stride_x,
               const uint16_t stride_y,
               const int32_t *bias,
               q7_t *output,
               const int32_t *output_shift,
               const int32_t *output_mult,
               const int32_t out_offset,
               const int32_t input_offset,
               const int32_t out_activation_min,
               const int32_t out_activation_max,
               const uint16_t output_x,
               const uint16_t output_y,
               q15_t *buffer_a);
void q7_to_q15_with_offset(const q7_t *src,
                               q15_t *dst,
                               uint32_t block_size,
                               q15_t offset);

q7_t *nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                    const q15_t *input_b,
                                    const uint16_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t out_offset,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const uint16_t num_col_a,
                                    const int32_t *const output_bias,
                                    q7_t *out_0);
arm_status max_pool_s8_opt(const uint16_t input_y,
                               const uint16_t input_x,
                               const uint16_t output_y,
                               const uint16_t output_x,
                               const uint16_t stride_y,
                               const uint16_t stride_x,
                               const uint16_t kernel_y,
                               const uint16_t kernel_x,
                               const uint16_t pad_y,
                               const uint16_t pad_x,
                               const int8_t act_min,
                               const int8_t act_max,
                               const uint16_t depth,
                               int8_t *src,
                               int16_t *tmp_buffer,
                               int8_t *dst);
static void compare_and_replace_if_larger_q7(q7_t *base,
                                             const q7_t *target,
                                             const uint16_t length);
static void clamp_output(q7_t *source, const uint16_t length, const int32_t act_min, const int32_t act_max);
arm_status
fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       q15_t *vec_buffer);
arm_status nn_vec_mat_mult_t_s8(const q7_t *lhs,
                                    const q7_t *rhs,
                                    const q31_t *bias,
                                    q7_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max);

// inference
arm_status cmsisRunInference (void* input, void* output)
{
  arm_status status;

  __disable_irq();

  for(int i = 0; i < CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH; i++) printf("%d, ", ((int8_t*)input)[i]);
  hiSendS8((int8_t*)input, CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH, 0);

  printf("\n// layer: conv1, input -> activations1\n");
  printf("required buf size: %d\n", arm_convolve_s8_get_buffer_size(CONV1_INPUT_CH, CONV1_KERNEL_X, CONV1_KERNEL_Y));
  
  status = convolve_s8((const q7_t *)input, CONV1_INPUT_X, CONV1_INPUT_Y, CONV1_INPUT_CH, CONV1_INPUT_BATCHES, 
    conv1_wt, CONV1_OUTPUT_CH, CONV1_KERNEL_X, CONV1_KERNEL_Y, 
    CONV1_PAD_X, CONV1_PAD_Y, CONV1_STRIDE_X, CONV1_STRIDE_Y,  conv1_bias, 
    (q7_t *)activations1, conv1_output_shift, conv1_output_mult, 
    CONV1_OUT_OFFSET, CONV1_INPUT_OFFSET, CONV1_OUTPUT_ACTIVATION_MIN, CONV1_OUTPUT_ACTIVATION_MAX, 
    CONV1_OUTPUT_X, CONV1_OUTPUT_Y, (q15_t *)tmpBuf);
  
  printf("status = %d\n", status);
  // for(int i = 0; i < CONV1_OUTPUT_CH*CONV1_OUTPUT_X*CONV1_OUTPUT_Y; i++) printf("%d, ", activations1[i]);
  hiSendS8((int8_t*)activations1, CONV1_OUTPUT_CH*CONV1_OUTPUT_X*CONV1_OUTPUT_Y, 1);


  printf("\n// layer: pool1, activations1 -> activations2\n");
  status = max_pool_s8_opt(POOL1_INPUT_Y, POOL1_INPUT_X, POOL1_OUTPUT_Y, POOL1_OUTPUT_X, POOL1_STRIDE_Y, POOL1_STRIDE_X, POOL1_KERNEL_Y, POOL1_KERNEL_X, POOL1_PAD_Y, POOL1_PAD_X, POOL1_ACT_MIN, POOL1_ACT_MAX, POOL1_DEPTH, (int8_t *)activations1, (int16_t *)tmpBuf, (int8_t *)activations2);
  printf("status = %d\n", status);
  // for(int i = 0; i < POOL1_DEPTH*POOL1_OUTPUT_X*POOL1_OUTPUT_Y; i++) printf("%d, ", activations2[i]);
  hiSendS8((int8_t*)activations2, POOL1_DEPTH*POOL1_OUTPUT_X*POOL1_OUTPUT_Y, 2);


  printf("\n// layer: conv2, activations2 -> activations1\n");
  printf("required buf size: %d\n", arm_convolve_s8_get_buffer_size(CONV2_INPUT_CH, CONV2_KERNEL_X, CONV2_KERNEL_Y));
  status = convolve_s8((const q7_t *)activations2,CONV2_INPUT_X, CONV2_INPUT_Y, CONV2_INPUT_CH, CONV2_INPUT_BATCHES, conv2_wt, CONV2_OUTPUT_CH, CONV2_KERNEL_X, CONV2_KERNEL_Y, CONV2_PAD_X, CONV2_PAD_Y, CONV2_STRIDE_X, CONV2_STRIDE_Y, conv2_bias, (q7_t *)activations1,conv2_output_shift, conv2_output_mult, CONV2_OUT_OFFSET, CONV2_INPUT_OFFSET, CONV2_OUTPUT_ACTIVATION_MIN, CONV2_OUTPUT_ACTIVATION_MAX, CONV2_OUTPUT_X, CONV2_OUTPUT_Y, (q15_t *)tmpBuf);
  printf("status = %d\n", status);
  // for(int i = 0; i < CONV2_OUTPUT_CH*CONV2_OUTPUT_X*CONV2_OUTPUT_Y; i++) printf("%d, ", activations1[i]);
  hiSendS8((int8_t*)activations1, CONV2_OUTPUT_CH*CONV2_OUTPUT_X*CONV2_OUTPUT_Y, 3);


  printf("\n// layer: pool2, activations1 -> activations2\n");
  status = max_pool_s8_opt(POOL2_INPUT_Y, POOL2_INPUT_X, POOL2_OUTPUT_Y, POOL2_OUTPUT_X, POOL2_STRIDE_Y, POOL2_STRIDE_X, POOL2_KERNEL_Y, POOL2_KERNEL_X, POOL2_PAD_Y, POOL2_PAD_X, POOL2_ACT_MIN, POOL2_ACT_MAX, POOL2_DEPTH, (int8_t *)activations1, (int16_t *)tmpBuf, (int8_t *)activations2);
  printf("status = %d\n", status);
  // for(int i = 0; i < POOL2_DEPTH*POOL2_OUTPUT_X*POOL2_OUTPUT_Y; i++) printf("%d, ",  activations2[i]);
  hiSendS8((int8_t*)activations2, POOL2_DEPTH*POOL2_OUTPUT_X*POOL2_OUTPUT_Y, 4);


  printf("\n// layer: conv3, activations2 -> activations1\n");
  printf("required buf size: %d\n", arm_convolve_s8_get_buffer_size(CONV3_INPUT_CH, CONV3_KERNEL_X, CONV3_KERNEL_Y));
  status = arm_convolve_s8((const q7_t *)activations2,CONV3_INPUT_X, CONV3_INPUT_Y, CONV3_INPUT_CH, CONV3_INPUT_BATCHES, conv3_wt, CONV3_OUTPUT_CH, CONV3_KERNEL_X, CONV3_KERNEL_Y, CONV3_PAD_X, CONV3_PAD_Y, CONV3_STRIDE_X, CONV3_STRIDE_Y, conv3_bias, (q7_t *)activations1,conv3_output_shift, conv3_output_mult, CONV3_OUT_OFFSET, CONV3_INPUT_OFFSET, CONV3_OUTPUT_ACTIVATION_MIN, CONV3_OUTPUT_ACTIVATION_MAX, CONV3_OUTPUT_X, CONV3_OUTPUT_Y, (q15_t *)tmpBuf);
  printf("status = %d\n", status);
  // for(int i = 0; i < CONV3_OUTPUT_CH*CONV3_OUTPUT_X*CONV3_OUTPUT_Y; i++) printf("%d, ", activations1[i] );
  hiSendS8((int8_t*)activations1, CONV3_OUTPUT_CH*CONV3_OUTPUT_X*CONV3_OUTPUT_Y, 5);


  printf("\n// layer: conv4, activations1 -> activations2\n");
  printf("required buf size: %d\n", arm_convolve_s8_get_buffer_size(CONV4_INPUT_CH, CONV4_KERNEL_X, CONV4_KERNEL_Y));
  status = arm_convolve_s8((const q7_t *)activations1,CONV4_INPUT_X, CONV4_INPUT_Y, CONV4_INPUT_CH, CONV4_INPUT_BATCHES, conv4_wt, CONV4_OUTPUT_CH, CONV4_KERNEL_X, CONV4_KERNEL_Y, CONV4_PAD_X, CONV4_PAD_Y, CONV4_STRIDE_X, CONV4_STRIDE_Y, conv4_bias, (q7_t *)activations2,conv4_output_shift, conv4_output_mult, CONV4_OUT_OFFSET, CONV4_INPUT_OFFSET, CONV4_OUTPUT_ACTIVATION_MIN, CONV4_OUTPUT_ACTIVATION_MAX, CONV4_OUTPUT_X, CONV4_OUTPUT_Y, (q15_t *)tmpBuf);
  printf("status = %d\n", status);
  // for(int i = 0; i < CONV4_OUTPUT_CH*CONV4_OUTPUT_X*CONV4_OUTPUT_Y; i++) printf("%d, ",  activations2[i]);
  hiSendS8((int8_t*)activations2, CONV4_OUTPUT_CH*CONV4_OUTPUT_X*CONV4_OUTPUT_Y, 6);


  printf("\n// layer: fc1, activations2 -> output\n");
  status = fully_connected_s8((const int8_t *)activations2, fc1_wt, FC1_COL_DIM, FC1_ROW_DIM, FC1_NB_BATCHES, FC1_INPUT_OFFSET, FC1_FILTER_OFFSET, FC1_OUT_MULT, FC1_OUT_SHIFT, FC1_OUTPUT_OFFSET, fc1_bias, (int8_t *)output, FC1_OUTPUT_ACTIVATION_MIN, FC1_OUTPUT_ACTIVATION_MAX, (q15_t *)tmpBuf);
  printf("status = %d\n", status);
  // for(int i = 0; i < FC1_ROW_DIM; i++) printf("%d, ", ((int8_t*)output)[i]);
  hiSendS8((int8_t*)output, FC1_ROW_DIM, 7);

  __enable_irq();

  return status;
}


arm_status convolve_s8(const q7_t *input,
               const uint16_t input_x,
               const uint16_t input_y,
               const uint16_t input_ch,
               const uint16_t input_batches,
               const q7_t *kernel,
               const uint16_t output_ch,
               const uint16_t kernel_x,
               const uint16_t kernel_y,
               const uint16_t pad_x,
               const uint16_t pad_y,
               const uint16_t stride_x,
               const uint16_t stride_y,
               const int32_t *bias,
               q7_t *output,
               const int32_t *output_shift,
               const int32_t *output_mult,
               const int32_t out_offset,
               const int32_t input_offset,
               const int32_t out_activation_min,
               const int32_t out_activation_max,
               const uint16_t output_x,
               const uint16_t output_y,
               q15_t *buffer_a)
{
  int i_batch;
  for (i_batch = 0; i_batch < input_batches; i_batch++)
  {
    int32_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    /* Generate two columns from the input tensor a GEMM computation */
    q15_t *two_column_buf = buffer_a;
    q7_t *out = output;

    /* This part implements the im2col function */
    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
      for (i_out_x = 0; i_out_x < output_x; i_out_x++)
      {
        for (i_ker_y = i_out_y * stride_y - pad_y; i_ker_y < i_out_y * stride_y - pad_y + kernel_y; i_ker_y++)
        {
          for (i_ker_x = i_out_x * stride_x - pad_x; i_ker_x < i_out_x * stride_x - pad_x + kernel_x; i_ker_x++)
          {
            if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x)
            {
              /* Filling 0 for out-of-bound paddings */
              memset(two_column_buf, 0, sizeof(q15_t) * input_ch);
            }
            else
            {
              /* Copying the pixel data to column */
              q7_to_q15_with_offset(input + (i_ker_y * input_x + i_ker_x) * input_ch, two_column_buf, input_ch, input_offset);
            }
            two_column_buf += input_ch;
          }
        }

        /* Computation is filed for every 2 columns */
        if (two_column_buf == buffer_a + 2 * input_ch * kernel_y * kernel_x)
        {
          out =
            nn_mat_mult_kernel_s8_s16(kernel,
                            buffer_a,
                            output_ch,
                            output_shift,
                            output_mult,
                            out_offset,
                            out_activation_min,
                            out_activation_max,
                            input_ch * kernel_y * kernel_x,
                            bias,
                            out);

          /* counter reset */
          two_column_buf = buffer_a;
        }
      }
    }

    /* left-over because odd number of output pixels */
    if (two_column_buf != buffer_a)
    {
      const q7_t *ker_a = kernel;
      int i;

      for (i = 0; i < output_ch; i++)
      {
        /* Load the accumulator with bias first */
        q31_t sum = bias[i];

        /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
        const q15_t *ip_as_col = buffer_a;

        /* 4 multiply and accumulates are done in one loop. */
        uint16_t col_count = (input_ch * kernel_y * kernel_x) >> 2;

        while (col_count)
        {
          q31_t ker_a1, ker_a2;
          q31_t ip_b1, ip_b2;

          ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

          ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
          sum = __SMLAD(ker_a1, ip_b1, sum);
          ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
          sum = __SMLAD(ker_a2, ip_b2, sum);

          col_count--;
        }
        /* Handle left over mac */
        col_count = input_ch * kernel_y * kernel_x & 0x3;
        while (col_count)
        {
          q7_t ker_a1 = *ker_a++;
          q15_t ip_b1 = *ip_as_col++;
          sum += ker_a1 * ip_b1;
          col_count--;
        }

        sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
        sum += out_offset;
        sum = MAX(sum, out_activation_min);
        sum = MIN(sum, out_activation_max);
        *out++ = (q7_t)sum;
      }
    }
    /* Advance to the next batch */
    input += (input_x * input_y * input_ch);
    output += (output_x * output_y * output_ch);
  }

  /* Return to application */
  return ARM_MATH_SUCCESS;
}

void q7_to_q15_with_offset(const q7_t *src,
                               q15_t *dst,
                               uint32_t block_size,
                               q15_t offset)
{
    int block_cnt;

    /* Run the below code for cores that support SIMD instructions  */
    q31_t in_q7x4;
    q31_t in_q15x2_1;
    q31_t in_q15x2_2;
    q31_t out_q15x2_1;
    q31_t out_q15x2_2;

    /*loop unrolling */
    block_cnt = block_size >> 2;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time. */
    const q31_t offset_q15x2 = __PKHBT(offset, offset, 16);
    while (block_cnt > 0)
    {
        /* convert from q7 to q15 and then store the results in the destination buffer */
        in_q7x4 = arm_nn_read_q7x4_ia(&src);

        /* Extract and sign extend each of the four q7 values to q15 */
        in_q15x2_1 = __SXTAB16(offset_q15x2, __ROR(in_q7x4, 8));
        in_q15x2_2 = __SXTAB16(offset_q15x2, in_q7x4);

        out_q15x2_2 = __PKHTB(in_q15x2_1, in_q15x2_2, 16);
        out_q15x2_1 = __PKHBT(in_q15x2_2, in_q15x2_1, 16);

        write_q15x2_ia(&dst, out_q15x2_1);
        write_q15x2_ia(&dst, out_q15x2_2);

        block_cnt--;
    }
    /* Handle left over samples */
    block_cnt = block_size % 0x4;

    while (block_cnt > 0)
    {
        *dst++ = (q15_t)*src++ + offset;

        /* Decrement the loop counter */
        block_cnt--;
    }
}

q7_t *nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                    const q15_t *input_b,
                                    const uint16_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t out_offset,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const uint16_t num_col_a,
                                    const int32_t *const output_bias,
                                    q7_t *out_0)
{
    /* set up the second output pointers */
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const q7_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        /* align the second pointer for A */
        const q7_t *ip_a1 = ip_a0 + num_col_a;

        /* Init accumulator with bias for channel N and N + 1 */
        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = *bias++;
        q31_t ch_1_out_0 = *bias;
        q31_t ch_1_out_1 = *bias++;

        uint16_t col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
            q31_t a01, a02, a11, a12;
            q31_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            q31_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad(ip_a1, &a11, &a12);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        } /* while over col_count */
        col_count = num_col_a & 0x3;
        while (col_count)
        {
            q7_t a0 = *ip_a0++;
            q15_t b0 = *ip_b0++;
            q7_t a1 = *ip_a1++;
            q15_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (q7_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (q7_t)ch_1_out_0;

        ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (q7_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        /* load the bias */
        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = *bias++;

        uint16_t col_count = num_col_a >> 2;
        while (col_count)
        {
            q31_t a01, a02;
            q31_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            q31_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = num_col_a & 0x3;
        while (col_count)
        {
            q7_t a0 = *ip_a0++;
            q15_t b0 = *ip_b0++;
            q15_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (q7_t)ch_0_out_1;
        out_mult++;
        out_shift++;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}
arm_status max_pool_s8_opt(const uint16_t input_y,
                               const uint16_t input_x,
                               const uint16_t output_y,
                               const uint16_t output_x,
                               const uint16_t stride_y,
                               const uint16_t stride_x,
                               const uint16_t kernel_y,
                               const uint16_t kernel_x,
                               const uint16_t pad_y,
                               const uint16_t pad_x,
                               const int8_t act_min,
                               const int8_t act_max,
                               const uint16_t depth,
                               int8_t *src,
                               int16_t *tmp_buffer,
                               int8_t *dst)
{

    /* Run the following code for Cortex-M4 and Cortex-M7 */
    (void)tmp_buffer;
    int32_t i_x, i_y;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < input_y; i_y++)
    {

        for (i_x = 0; i_x < output_x; i_x++)
        {
            /* for each output sample */
            q7_t *target = src + (i_y * input_x + i_x) * depth;
            q7_t *win_start;
            q7_t *win_stop;
            const int32_t x_origin = i_x * stride_x - pad_x;
            if (x_origin < 0)
            {
                win_start = target;
            }
            else
            {
                win_start = src + (i_y * input_x + x_origin) * depth;
            }

            if (x_origin + kernel_x >= input_x)
            {
                win_stop = src + (i_y * input_x + input_x) * depth;
            }
            else
            {
                win_stop = src + (i_y * input_x + x_origin + kernel_x) * depth;
            }

            /* first step is to copy over initial data(along channel) along the channel in  x direction */
            memmove(target, win_start, depth);

            /* Move over to next element along x axis and compare with the base(target)  */
            win_start += depth;
            for (; win_start < win_stop; win_start += depth)
            {
                compare_and_replace_if_larger_q7(target, win_start, depth);
            }
        }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < output_y; i_y++)
    {
        /* for each output row */
        q7_t *target = dst + i_y * output_x * depth;
        q7_t *row_start;
        q7_t *row_end;
        const int32_t y_origin = i_y * stride_y - pad_y;
        /* setting the starting row */
        if (y_origin < 0)
        {
            row_start = src;
        }
        else
        {
            row_start = src + y_origin * input_x * depth;
        }
        /* setting the stopping row */
        if (y_origin + kernel_y >= input_y)
        {
            row_end = src + input_y * input_x * depth;
        }
        else
        {
            row_end = src + (y_origin + kernel_y) * input_x * depth;
        }

        /* copy over the complete first row. */
        memmove(target, row_start, output_x * depth);

        /* move over to next row and compare with the base row (target)*/
        row_start += depth * input_x;

        for (; row_start < row_end; row_start += input_x * depth)
        {
            compare_and_replace_if_larger_q7(target, row_start, output_x * depth);
        }
    }

    clamp_output(dst, output_x * output_y * depth, act_min, act_max);

  return ARM_MATH_SUCCESS;
}
static void compare_and_replace_if_larger_q7(q7_t *base,
                                             const q7_t *target,
                                             const uint16_t length)
{
    q7_t *dst = base;
    const q7_t *src = target;
    union arm_nnword ref_max;
    union arm_nnword comp_max;
    int32_t cnt = length >> 2;

    while (cnt > 0l)
    {
        ref_max.word = arm_nn_read_q7x4(dst);
        comp_max.word = arm_nn_read_q7x4_ia(&src);

        if (comp_max.bytes[0] > ref_max.bytes[0])
        {
            ref_max.bytes[0] = comp_max.bytes[0];
        }
        if (comp_max.bytes[1] > ref_max.bytes[1])
        {
            ref_max.bytes[1] = comp_max.bytes[1];
        }
        if (comp_max.bytes[2] > ref_max.bytes[2])
        {
            ref_max.bytes[2] = comp_max.bytes[2];
        }
        if (comp_max.bytes[3] > ref_max.bytes[3])
        {
            ref_max.bytes[3] = comp_max.bytes[3];
        }

        write_q7x4_ia(&dst, ref_max.word);

        cnt--;
    }

    cnt = length & 0x3;
    while (cnt > 0l)
    {
        if (*src > *dst)
        {
            *dst = *src;
        }
        dst++;
        src++;
        cnt--;
    }
}
static void clamp_output(q7_t *source, const uint16_t length, const int32_t act_min, const int32_t act_max)
{
    union arm_nnword in;
    int32_t cnt = length >> 2;

    while (cnt > 0l)
    {
        in.word = arm_nn_read_q7x4(source);

        in.bytes[0] = MAX(in.bytes[0], act_min);
        in.bytes[0] = MIN(in.bytes[0], act_max);
        in.bytes[1] = MAX(in.bytes[1], act_min);
        in.bytes[1] = MIN(in.bytes[1], act_max);
        in.bytes[2] = MAX(in.bytes[2], act_min);
        in.bytes[2] = MIN(in.bytes[2], act_max);
        in.bytes[3] = MAX(in.bytes[3], act_min);
        in.bytes[3] = MIN(in.bytes[3], act_max);

        write_q7x4_ia(&source, in.word);
        cnt--;
    }

    cnt = length & 0x3;
    while (cnt > 0l)
    {
        int32_t comp = *source;
        comp = MAX(comp, act_min);
        comp = MIN(comp, act_max);
        *source++ = (int8_t)comp;
        cnt--;
    }
}
arm_status
fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       q15_t *vec_buffer)
{

    (void)vec_buffer;

    int32_t batch_cnt = nb_batches;

    while (batch_cnt)
    {
        nn_vec_mat_mult_t_s8(input,
                                 kernel,
                                 bias,
                                 output,
                                 input_offset,
                                 filter_offset,
                                 output_offset,
                                 out_mult,
                                 out_shift,
                                 col_dim,
                                 row_dim,
                                 output_activation_min,
                                 output_activation_max);
        input += col_dim;
        output += row_dim;
        batch_cnt--;
    }
    return (ARM_MATH_SUCCESS);
}
arm_status nn_vec_mat_mult_t_s8(const q7_t *lhs,
                                    const q7_t *rhs,
                                    const q31_t *bias,
                                    q7_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{

    const int32_t off0 = rhs_cols - 4;
    const int16_t lhs_offset_s16 = lhs_offset;
    const int16_t rhs_offset_s16 = rhs_offset;

    const uint32_t lhs_offset_s16x2 = __PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    const uint32_t rhs_offset_s16x2 = __PKHBT(rhs_offset_s16, rhs_offset_s16, 16);

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;
        q31_t res01 = *bias++;

        int32_t rhs_cols_idx = 0;

        q31_t val0, val1, val2, val3, val4, val5;
        for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
        {
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);
        }

        for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t rhs_value1 = rhs_ptr[rhs_cols] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);

        *dst++ = (q7_t)res00;
        *dst++ = (q7_t)res01;

        rhs += 2 * rhs_cols;
    }

    if (rhs_rows % 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;

        int32_t rhs_cols_idx = 0;

        q31_t val0, val1, val2, val3;
        for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
        {
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);
        }

        for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (q7_t)res00;
    }

    return ARM_MATH_SUCCESS;
}