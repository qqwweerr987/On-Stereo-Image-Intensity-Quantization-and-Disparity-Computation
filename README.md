# Disparity Computation on Low Intensity Quantization
This Code is for CVPR 2024 submission 

From variate bit-rate stereo matching, it is observed that the image pair with a low intensity quantization level is still capable of providing good disparity maps. In this paper, we further justify the fact for deep learning-based stereo algorithms. Moreover, a mathematical model representing the level of disparity discontinuity is proposed to formulate the mismatching prediction based on the intensity quantization. It is then used to derive the minimum quantization level for quality-assured stereo matching. Due to the high computational cost of stereo processing, the reduction of image data usage will benefit both network training and real-time inference. The formulation presented in this work is investigated extensively for various types of scenes for disparity computation. In the experiments, the feasibility of our approach is validated with the real scene Middlebury, KITTI 2015, and synthetic Scene Flow datasets. 

By exploiting the mathematical model and algorithms related stereo matching and image intensity quantization, the disparity error on different bit-rate image pairs is predicted.
The proposed formulation explores the possibility to simultaneously maintain the stereo matching performance and reduce the data for processing. It can serve as a framework for
the stereo matching algorithms, 
cluding the deep learningbased approaches, to alleviate the computation requirement.

in![1105](https://github.com/qqwweerr987/stereo-quantization-disparity/assets/45920949/86636442-ed39-4d89-bcd3-e3a8a4ae5b7c)
#### The disparity maps of ‘Jadeplant’ obtained using RAFTstereo on the image pairs with 2, 4 and 256 intensity quantization levels (from the left to the right).

Since the intensity quantization is an intuitive way for data compression, the idea can be implemented with the existing techniques. The enduring problem of memory efficiency on
stereo matching can also be mitigated without the expensive hardware.

In this work, we propose the idea of disparity discontinuity level (DDL) to predict the disparity error under different intensity quantization. It can be considered as a metric to represent the disparity mismatch due to the difference between left and right images caused by feature disappearance during quantization.

Run **OnStereoError.py** to get the eight value of predicted error.

Note that the computaion on parameters *Interval* and *Threshold* are not contained in the code.
Please follow the approaches mentioned in the paper to compute the suitable parameters.
Image from public datasets Middlebury / KITTI2015 / Sceneflow used in our research are placed in the folder.  
