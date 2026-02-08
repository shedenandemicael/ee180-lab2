#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include <arm_neon.h>
using namespace cv;


/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
void sobelCalc(Mat& img, Mat& img_sobel_out)
{
  // double color;
  static Mat img_gray;
  
  img_gray = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j+=4) {
      float32x4_t scalar;

      uint8x8x3_t data = vld3_u8(&img.data[STEP0*i + STEP1*j]);
      float32x4_t data1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(data.val[0]))));
      float32x4_t data2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(data.val[1]))));
      float32x4_t data3 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(data.val[2]))));
      scalar = vdupq_n_f32(.114f);
      float32x4_t op1 = vmulq_f32(data1, scalar);
      scalar = vdupq_n_f32(.587f);
      float32x4_t op2 = vmulq_f32(data2, scalar);
      scalar = vdupq_n_f32(.299f);
      float32x4_t op3 = vmulq_f32(data3, scalar);
      float32x4_t colorfp = vaddq_f32(vaddq_f32(op1, op2), op3);
      uint16x4_t coloru = vqmovn_u32(vcvtq_u32_f32(colorfp));
      uint8x8_t color = vqmovn_u16(vcombine_u16(coloru, coloru));

      vst1_u8(&img_gray.data[IMG_WIDTH*i + j], color);

      // color = .114*img.data[STEP0*i + STEP1*j] +
      //         .587*img.data[STEP0*i + STEP1*j + 1] +
      //         .299*img.data[STEP0*i + STEP1*j + 2];
      // img_gray.data[IMG_WIDTH*i + j] = color;
    }
  }

  // Apply Sobel filter to black & white image
  unsigned short sobel;
  unsigned short sobelx;
  unsigned short sobely;

  // Calculate the x and y convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobelx = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobely = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		  2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		  2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      // Combine the two convolutions into the output image
      sobel = sobelx + sobely;
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }


  // double color;
  // unsigned short sobel;
  // unsigned short sobelx;
  // unsigned short sobely;
  // static Mat gray_buf;

  // // initialize gray buffer
  // gray_buf = Mat(3, IMG_WIDTH, CV_8UC1);
  // unsigned short row_start = 0;
  // for (int i=0; i<3; i++) {
  //   for (int j=0; j<img.cols; j++) {
  //     color = .114*img.data[STEP0*i + STEP1*j] +
  //             .587*img.data[STEP0*i + STEP1*j + 1] +
  //             .299*img.data[STEP0*i + STEP1*j + 2];
  //     gray_buf.data[IMG_WIDTH*i + j] = color;
  //   }
  // }

  // for (int i=1; i<img.rows; i++) {
  //   for (int j=1; j<img.cols; j++) {
  //     sobelx = abs(gray_buf.data[IMG_WIDTH*((row_start+2)%3) + (j-1)] -
  //       gray_buf.data[IMG_WIDTH*((row_start+1)%3) + (j-1)] +
  //       2*gray_buf.data[IMG_WIDTH*((row_start+2)%3) + (j)] -
  //       2*gray_buf.data[IMG_WIDTH*((row_start+1)%3) + (j)] +
  //       gray_buf.data[IMG_WIDTH*((row_start+2)%3) + (j+1)] -
  //       gray_buf.data[IMG_WIDTH*((row_start+1)%3) + (j+1)]);

  //     sobely = abs(gray_buf.data[IMG_WIDTH*((row_start+2)%3) + (j-1)] -
  //       gray_buf.data[IMG_WIDTH*((row_start+2)%3) + (j+1)] +
  //       2*gray_buf.data[IMG_WIDTH*(row_start%3) + (j-1)] -
  //       2*gray_buf.data[IMG_WIDTH*(row_start%3) + (j+1)] +
  //       gray_buf.data[IMG_WIDTH*((row_start+1)%3) + (j-1)] -
  //       gray_buf.data[IMG_WIDTH*((row_start+1)%3) + (j+1)]);

  //     // Combine the two convolutions into the output image
  //     sobel = sobelx + sobely;
  //     sobel = (sobel > 255) ? 255 : sobel;
  //     img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
  //   }
  //   for (int j=0; j<img.cols; j++) {
  //     color = .114*img.data[STEP0*row_start + STEP1*j] +
  //           .587*img.data[STEP0*row_start + STEP1*j + 1] +
  //           .299*img.data[STEP0*row_start + STEP1*j + 2];
  //     gray_buf.data[IMG_WIDTH*row_start + j] = color;
  //   }
  //   row_start = (row_start + 1) % 3;
  // }
}
