This Code is for ICCV 2023 submission 
### Disparity Computation on Low Intensity Quantization



Run **OnStereoError.py** to get the eight value of predicted error.

Note that the computaion on parameters *Interval* and *Threshold* are not contained in the code.

Please follow the approaches mentioned in the paper to compute the suitable parameters.

Image from public datasets Middlebury / KITTI2015 / Sceneflow used in our research are placed in the folder.  

From variate bit-rate stereo matching, it is observed that the image pair with a low intensity quantization level is still capable of providing good disparity maps. In this paper, we further justify the fact for deep learning-based stereo algorithms. Moreover, a mathematical model representing the level of disparity discontinuity is proposed to formulate the mismatching prediction based on the intensity quantization. It is then used to derive the minimum quantization level for quality-assured stereo matching. Due to the high computational cost of stereo processing, the reduction of image data usage will benefit both network training and real-time inference. The formulation presented in this work is investigated extensively for various types of scenes for disparity computation. In the experiments, the feasibility of our approach is validated with the real scene Middlebury, KITTI 2015, and synthetic Scene Flow datasets. The source code is available at https://github.com/qqwweerr987/stereo-quantization-disparity
