#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
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
  double color;

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j++) {
      color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img.data[IMG_WIDTH*i + j] = color;
    }
  }

  // Apply Sobel filter to black & white image
  unsigned short sobel;
  unsigned short sobelx;
  unsigned short sobely;

  // Calculate the x and y convolution
  for (int i=1; i<img.rows; i++) {
    for (int j=1; j<img.cols; j++) {
      sobelx = abs(img.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img.data[IMG_WIDTH*(i+1) + (j)] +
		  img.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobelx = (sobelx > 255) ? 255 : sobelx;

      sobely = abs(img.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img.data[IMG_WIDTH*(i-1) + (j+1)] +
		  2*img.data[IMG_WIDTH*(i) + (j-1)] -
		  2*img.data[IMG_WIDTH*(i) + (j+1)] +
		  img.data[IMG_WIDTH*(i+1) + (j-1)] -
		  img.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobely = (sobely > 255) ? 255 : sobely;

      // Combine the two convolutions into the output image
      sobel = sobelx + sobely;
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }
}
