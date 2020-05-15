#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/QR>
#include "homography.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core/eigen.hpp>
#include <boost/format.hpp>
void decomposePMatrix() {
    using namespace Eigen;

    MatrixXd pMatrix(3, 4);
    pMatrix(0, 0) = 3.53553e+2;
    pMatrix(0, 1) = 3.39645e+2;
    pMatrix(0, 2) = 2.77744e+2;
    pMatrix(0, 3) = -1.44946e+6;
    pMatrix(1, 0) = -1.03528e+2;
    pMatrix(1, 1) = 2.33212e+1;
    pMatrix(1, 2) = 4.59607e+2;
    pMatrix(1, 3) = -6.32525e+5;
    pMatrix(2, 0) = 7.07107e-1;
    pMatrix(2, 1) = -3.53553e-1;
    pMatrix(2, 2) = 6.12372e-1;
    pMatrix(2, 3) = -9.18559e+2;

    Matrix3d M = pMatrix.block(0, 0, 3, 3);
    HouseholderQR<Matrix3d> householderQR;;

    householderQR.compute(M.inverse());

    MatrixXd Q = householderQR.householderQ();
    MatrixXd R =  householderQR.matrixQR().triangularView<Upper>();


    MatrixXd K = R.inverse().cwiseAbs(); // 绝对值

    MatrixXd Rotate =  K.inverse() * M;
    MatrixXd T = K.inverse() * pMatrix.col(3);
    MatrixXd C = -Rotate.transpose()*T;

    std::cout <<"C " << C << std::endl;
    std::cout << "Hello, world!" << std::endl;
    std::cout << "K " << K << std::endl;
    std::cout << "Rotate " << Rotate << std::endl;
    std::cout << "R " << Q.transpose() << std::endl;
    std::cout << "T " << T << std::endl;
}

void testHomography(){
    
     std::vector<std::string> files= {
        "/home/atway/code/opencv_data/left01.jpg",
        "/home/atway/code/opencv_data/left04.jpg"
    };

    std::vector<std::vector<cv::Point2f>> allCorners;
    cv::Size boardSize(6, 9);
    for(int i=0; i<files.size(); ++i) {

        cv::Mat img =  cv::imread(files[i]);
        std::vector<cv::Point2f> corners;

        bool ok = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(ok) {
            allCorners.push_back(corners);
           // cv::drawChessboardCorners(img, boardSize, cv::Mat(corners), ok);
           // cv::imshow("corners", img);
           // cv::waitKey();
        }

    }


    std::vector<Eigen::Vector2d> srcPoints, dstPoints;

    for(cv::Point2f& pt: allCorners[0])
    {
        srcPoints.push_back(Eigen::Vector2d(pt.x, pt.y));
    }
    for(cv::Point2f& pt: allCorners[1])
    {
        
        dstPoints.push_back(Eigen::Vector2d(pt.x, pt.y));

    }
    std::cout << srcPoints.size() << std::endl;
    Eigen::Matrix3d H;
    bool ok = findHomography(srcPoints, dstPoints, H, true);

    if(ok){
        std::cout << "H = " << H << std::endl;
        cv::Mat srcImage = cv::imread(files[0]);
        cv::Mat dstImage = cv::imread(files[1]);
        cv::Mat result;
        cv::Mat Hmat;
        cv::eigen2cv(H, Hmat);
        cv::warpPerspective(srcImage, result, Hmat, dstImage.size());
        cv::imshow("result", result);
        cv::waitKey();
        cv::imwrite("result.jpg", result);
    }
    /*
     * 优化前
     * H = 1.07368    -0.209543     -74.3206
        -0.0632842      1.02258      41.3408
        -0.00012842 -0.000406687            1
        
        // 优化后
        H = 1.07926    -0.155123      -74.276
        -0.0604588      1.02924      40.2497
        -0.000114997 -0.000406556            1
        
        // 归一化
        
        H =  1.04498    -0.221204     -59.6102
            -0.0475147     0.956021      42.3854
            -0.000110262 -0.000579577            1
     * 
    H =  1.59953   -0.0338682     -1059.74
        -0.033448       1.6032     -889.907
        -3.29594e-07 -1.55556e-06       1.0004
     
     */
    
}

/**
 * calibration
 */
void testCalibration(){
      std::vector<std::string> files = {
        "/home/atway/code/opencv_data/left01.jpg",
        "/home/atway/code/opencv_data/left02.jpg",
        "/home/atway/code/opencv_data/left03.jpg",
        "/home/atway/code/opencv_data/left04.jpg",
        "/home/atway/code/opencv_data/left05.jpg",
        "/home/atway/code/opencv_data/left06.jpg",
        "/home/atway/code/opencv_data/left07.jpg",
        "/home/atway/code/opencv_data/left08.jpg",
        "/home/atway/code/opencv_data/left09.jpg",
        "/home/atway/code/opencv_data/left11.jpg",
        "/home/atway/code/opencv_data/left12.jpg",
        "/home/atway/code/opencv_data/left13.jpg",
        "/home/atway/code/opencv_data/left14.jpg",
    };

    //files.resize(0);
    
    //for(int i=1; i<=10; ++i){
    //boost::format fmt("%s/image%d.%s");
    // files.push_back( (fmt % "/home/atway/code/slam/MVGAlgorithm/smartphoneimage" % i % "png").str());   
    //}
    
    std::vector<std::vector<Eigen::Vector2d>> imagePoints;
    std::vector<std::vector<Eigen::Vector3d>> objectPoints;
    cv::Size boardSize(9, 6);
    cv::Size2f squareSize(25., 25.);
    for(int i=0; i<files.size(); ++i) {

        cv::Mat img =  cv::imread(files[i]);
        std::vector<cv::Point2f> corners;

        bool ok = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(ok) {
            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            
            
           cv::drawChessboardCorners(img, boardSize, cv::Mat(corners), ok);
           cv::imshow("corners", img);
           cv::waitKey(100);
            std::vector<Eigen::Vector2d> _corners;
            for(auto& pt: corners){
                _corners.push_back(Eigen::Vector2d(pt.x, pt.y));
            }
            imagePoints.push_back(_corners);
        }

    }
    
    for(int i=0; i<imagePoints.size(); ++i){
        std::vector<Eigen::Vector3d> corners;
        getObjecPoints(boardSize, squareSize, corners);
        objectPoints.push_back(corners);
    }
    
    
    computeCameraCalibration(imagePoints, objectPoints);
    
    
    /*
     * 
     * 536.073 536.016 342.371 235.536 
     * -0.265091 -0.0467179 0.252214 
     *  0.00183296 -0.000314465  
     * avg re projection error = 0.234593
        0 projection error = 0.16992
        1 projection error = 0.846329
        2 projection error = 0.159117
        3 projection error = 0.176626
        4 projection error = 0.141207
        5 projection error = 0.162312
        6 projection error = 0.18801
        7 projection error = 0.214098
        8 projection error = 0.222171
        9 projection error = 0.153192
        10 projection error = 0.177543
        11 projection error = 0.28586
        12 projection error = 0.15332
        
        
        opencv 的结果
        camera_matrix: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [ 5.3591573396163199e+02, 0., 3.4228315473308373e+02, 0.,
            5.3591573396163199e+02, 2.3557082909788173e+02, 0., 0., 1. ]
        distortion_coefficients: !!opencv-matrix
        rows: 5
        cols: 1
        dt: d
        data: [ -2.6637260909660682e-01, -3.8588898922304653e-02,
            1.7831947042852964e-03, -2.8122100441115472e-04,
            2.3839153080878486e-01 ]
        avg_reprojection_error: 3.9259098975581364e-01
     * 
     */
}
int main(int argc, char **argv) {

  
    
    testCalibration();

  
    return 0;
}
