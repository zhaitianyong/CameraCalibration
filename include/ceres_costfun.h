#ifndef CERES_COSTFUN_H
#define CERES_COSTFUN_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct HOMOGRAPHY_COST {

    HOMOGRAPHY_COST(double x1, double y1,double x2, double y2)
        :x1(x1), y1(y1), x2(x2), y2(y2)
    {}

    template<typename T>
    bool operator()(const T *const h, T* residual) const
    {

        T w = T(h[6]*x1+h[7]*y1+h[8]);
        T x = T(h[0]*x1+h[1]*y1+h[2])/w;
        T y = T(h[3]*x1+h[4]*y1+h[5])/w;

        residual[0] = ceres::sqrt(ceres::pow(T(x2)-x, 2) + ceres::pow(T(y2)-y, 2));

        return true;
    }

    const double x1, x2, y1, y2;

};


struct PROJECT_COST {

    Eigen::Vector3d objPt;
    Eigen::Vector2d imgPt;

    PROJECT_COST(Eigen::Vector3d& objPt, Eigen::Vector2d& imgPt):objPt(objPt), imgPt(imgPt)
    {}


    template<typename T>
    bool operator()(
        const T *const k,
        const T *const r,
        const T *const t,
        T* residuals)const
    {

        T pos3d[3] = {T(objPt(0)), T(objPt(1)), T(objPt(2))};
        T pos3d_proj[3];
        // 旋转
        ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
        // 平移
        pos3d_proj[0] += t[0];
        pos3d_proj[1] += t[1];
        pos3d_proj[2] += t[2];

        T xp = pos3d_proj[0] / pos3d_proj[2];
        T yp = pos3d_proj[1] / pos3d_proj[2];



        const T& fx = k[0];
        const T& fy = k[1];
        const T& cx = k[2];
        const T& cy = k[3];

        const T& k1 = k[4];
        const T& k2 = k[5];
        const T& k3 = k[6];

        const T& p1 = k[7];
        const T& p2 = k[8];

        T r_2 = xp*xp + yp*yp;

        /*
        // 径向畸变
        T xdis = xp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2);
        T ydis = yp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2);

        // 切向畸变
        xdis = xdis + T(2.)*p1*xp*yp + p2*(r_2 + T(2.)*xp*xp);
        ydis = ydis + p1*(r_2 + T(2.)*yp*yp) + T(2.)*p2*xp*yp;
        */

        // 径向畸变和切向畸变
        T xdis = xp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2) + T(2.)*p1*xp*yp + p2*(r_2 + T(2.)*xp*xp);
        T ydis = yp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2) + p1*(r_2 + T(2.)*yp*yp) + T(2.)*p2*xp*yp;
        

        // 像素距离
        T u = fx*xdis + cx;
        T v = fy*ydis + cy;

        residuals[0] = u - T(imgPt[0]);

        residuals[1] = v - T(imgPt[1]);

        return true;
    }

};

struct FISHER_PROJECT_COST {

    Eigen::Vector3d objPt;
    Eigen::Vector2d imgPt;

    FISHER_PROJECT_COST(Eigen::Vector3d& objPt, Eigen::Vector2d& imgPt):objPt(objPt), imgPt(imgPt)
    {}


    template<typename T>
    bool operator()(
            const T *const k,
            const T *const r,
            const T *const t,
            T* residuals)const
    {

        T pos3d[3] = {T(objPt(0)), T(objPt(1)), T(objPt(2))};
        T pos3d_proj[3];
        // 旋转
        ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
        // 平移
        pos3d_proj[0] += t[0];
        pos3d_proj[1] += t[1];
        pos3d_proj[2] += t[2];

        T xp = pos3d_proj[0] / pos3d_proj[2];
        T yp = pos3d_proj[1] / pos3d_proj[2];



        const T& fx = k[0];
        const T& fy = k[1];
        const T& cx = k[2];
        const T& cy = k[3];

        const T& k1 = k[4];
        const T& k2 = k[5];
        const T& k3 = k[6];
        const T& k4 = k[7];


        // 径向畸变
        T r_ = ceres::sqrt(xp * xp + yp * yp);

        // 等距离模型
        T theta = ceres::atan(r_);

        T thera_hat = theta * ( T(1.0) + k1*ceres::pow(theta, 2) + k2*ceres::pow(theta, 4) + k3*ceres::pow(theta, 6) + k4*ceres::pow(theta, 8));

        T xdis = thera_hat*xp / r_;
        T ydis = thera_hat*yp / r_;

        // 像素距离
        T u = fx*xdis + cx;
        T v = fy*ydis + cy;


        residuals[0] = u - T(imgPt[0]);

        residuals[1] = v - T(imgPt[1]);


        return true;
    }

};


#endif
