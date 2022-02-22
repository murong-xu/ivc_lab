IVC Lab 2020
Final Optimization
===========================
Last updated: 2020/08/14

##########Author##########
name: Murong Xu
mail: murong.xu@tum.de


##########Main Function##########
By running this project, six video compression methods including the two previous baselines will be performed.

1, Chp4 Baseline: Intra Coding
2, Chp5 Baseline: Video Coding
3, Intra Coding Optimization: Intra Prediction
4, Intra Coding Optimization: Adaptive Post-deblocking
5, Video Coding Optimization: Half-pel Motion Estimation
6, Video Coding Optimization: Quarter-pel Motion Estimation

The generated results will be stored in "result.mat" and plotted. The execution time will be displayed in command window.


##########Requirement##########
Install the Matlab Image Processing Toolbox. 
The code is tested on Matlab 2020a, with operating system Unix.


##########Usage Description##########
1. Unzip the file.
2. Open the 'main.m' in Matlab.
3. Run the 'main.m' with specifying the following input parameters at top of the script:
   * videoFolder    : path of the video folder
   * lena_small     : path of the small lena image
   * scales         : (default) quantization scales
4. Enjoy and have fun! 

