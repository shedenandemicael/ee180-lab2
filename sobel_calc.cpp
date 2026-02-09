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
  static Mat img_gray;
  
  img_gray = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j+=8) {
      uint8x8x3_t data = vld3_u8(&img.data[STEP0*i + STEP1*j]);
      uint16x8_t data1 = vmovl_u8(data.val[0]);
      uint16x8_t data2 = vmovl_u8(data.val[1]);
      uint16x8_t data3 = vmovl_u8(data.val[2]);
      uint8x8_t op1 = vshrn_n_u16(vmulq_n_u16(data1, 29), 8);  // multiply by 29/256 ~ .114
      uint8x8_t op2 = vshrn_n_u16(vmulq_n_u16(data2, 150), 8);  // multiply by 150/256 ~ .587
      uint8x8_t op3 = vshrn_n_u16(vmulq_n_u16(data3, 77), 8);  // multiply by 77/256 ~ .299
      uint8x8_t color = vadd_u8(vadd_u8(op1, op2), op3);

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
