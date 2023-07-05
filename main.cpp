#include<iostream>
#include<opencv2/opencv.hpp>
#include<ceres/ceres.h>
#include<ceres/rotation.h>

//#define DEBUG


bool readImages(const std::string& name, std::vector<cv::Mat>& images)
{
	// 查找路径下所有png图片，非递归。
	std::vector<std::string> list;
	cv::glob(name, list);

	if (list.empty())
		return false;
	else
	{
		std::cout << "找到图片数量：" << list.size() << std::endl;
		for (size_t i = 0; i < list.size(); ++i)
		{
			std::cout << list[i] << std::endl;
			// 读取图片至vector<Mat>
			images.push_back(cv::imread(list[i], cv::IMREAD_UNCHANGED));
		}
		return true;
	}
}

bool findCorners(const std::vector<cv::Mat>& images, const uint& w, const uint& h, const double& scale, std::vector<std::vector<cv::Point2f>>& imgPntsVec, std::vector<std::vector<cv::Point3f>>& objPntsVec)
{
	std::vector<cv::Point2f> imgPoints;
	std::vector<cv::Point3f> objPoints;
	for (size_t i = 0; i < h; ++i)
	{
		for (size_t j = 0; j < w; ++j)
		{
			objPoints.push_back(cv::Point3f(i * scale, j * scale, 0));
		}
	}

	size_t foundnumber = 0;
	for (uint i = 0; i < images.size(); ++i)
	{
		bool found = cv::findChessboardCornersSB(images[i], cv::Size(w, h), imgPoints, 0);
		if (found)
		{
			imgPntsVec.push_back(imgPoints);
			objPntsVec.push_back(objPoints);
			foundnumber++;
			// 定义DEBUG宏，可以显示棋盘格检测结果
			#ifdef DEBUG
			cv::namedWindow("查找到的角点");
			cv::drawChessboardCorners(images[i], cv::Size(w, h), imgPoints, true);
			cv::imshow("查找到的角点",images[i]);
			cv::waitKey();
			cv::destroyWindow("查找到的角点");
			#endif
		}
	}

	if (foundnumber < 3)
	{
		std::cout << "检测出的有效棋盘格标定板数量小于3！" << std::endl;
		return false;
	}
	else
	{
		std::cout << "检测出有效棋盘格标定板数量为：" << foundnumber << std::endl;
		return true;
	}
}

cv::Mat findHomography(const std::vector<cv::Point2f>& imgPoints, const std::vector<cv::Point3f>& objPoints)
{
	cv::Mat homo;
	const size_t number = imgPoints.size();
	cv::Mat A(number * 2, 9, CV_64F);
	for (size_t i = 0; i < number; ++i)
	{
		double u = imgPoints[i].x;
		double v = imgPoints[i].y;
		double x = objPoints[i].x;
		double y = objPoints[i].y;

		A.at<double>(2 * i, 0) = x;
		A.at<double>(2 * i, 1) = y;
		A.at<double>(2 * i, 2) = 1;
		A.at<double>(2 * i, 3) = 0;
		A.at<double>(2 * i, 4) = 0;
		A.at<double>(2 * i, 5) = 0;
		A.at<double>(2 * i, 6) = -u * x;
		A.at<double>(2 * i, 7) = -u * y;
		A.at<double>(2 * i, 8) = -u;

		A.at<double>(2 * i + 1, 0) = 0;
		A.at<double>(2 * i + 1, 1) = 0;
		A.at<double>(2 * i + 1, 2) = 0;
		A.at<double>(2 * i + 1, 3) = x;
		A.at<double>(2 * i + 1, 4) = y;
		A.at<double>(2 * i + 1, 5) = 1;
		A.at<double>(2 * i + 1, 6) = -v * x;
		A.at<double>(2 * i + 1, 7) = -v * y;
		A.at<double>(2 * i + 1, 8) = -v;
	}
	cv::SVD::solveZ(A, homo);
#ifdef DEBUG
	std::cout << "单应矩阵：" << homo.t() << std::endl;
#endif
	//homo = homo / homo.at<double>(8);
	if (homo.at<double>(8) < 0)
		homo = -homo;
	homo = homo.reshape(0, 3);

	return homo;
}

void findAllHomography(const std::vector<std::vector<cv::Point2f>>& imgPntsVec, const std::vector<std::vector<cv::Point3f>>& objPntsVec, std::vector<cv::Mat>& homoVec)
{
	for (size_t i = 0; i < imgPntsVec.size(); ++i)
	{
		homoVec.push_back(findHomography(imgPntsVec[i], objPntsVec[i]));
	}
}

void estimateIntrinsics(const std::vector<cv::Mat>& homoVec, cv::Mat K)
{

	cv::Mat v(homoVec.size() * 2, 5, CV_64F);
	for (int i = 0; i < homoVec.size(); ++i)
	{
		double h11 = homoVec[i].at<double>(0, 0);
		double h12 = homoVec[i].at<double>(1, 0);
		double h13 = homoVec[i].at<double>(2, 0);
		double h21 = homoVec[i].at<double>(0, 1);
		double h22 = homoVec[i].at<double>(1, 1);
		double h23 = homoVec[i].at<double>(2, 1);

		v.at<double>(i * 2, 0) = h11 * h21;
		v.at<double>(i * 2, 1) = h12 * h22;
		v.at<double>(i * 2, 2) = h13 * h21 + h11 * h23;
		v.at<double>(i * 2, 3) = h13 * h22 + h12 * h23;
		v.at<double>(i * 2, 4) = h13 * h23;

		v.at<double>(i * 2 + 1, 0) = h11 * h11 - h21 * h21;
		v.at<double>(i * 2 + 1, 1) = h12 * h12 - h22 * h22;
		v.at<double>(i * 2 + 1, 2) = h13 * h11 + h11 * h13 - h23 * h21 - h21 * h23;
		v.at<double>(i * 2 + 1, 3) = h13 * h12 + h12 * h13 - h23 * h22 - h22 * h23;
		v.at<double>(i * 2 + 1, 4) = h13 * h13 - h23 * h23;
	}
	cv::Mat B;
	cv::SVD::solveZ(v, B);
	std::cout << "B:" << B.t() << std::endl;

	double B11 = B.at<double>(0);
	double B22 = B.at<double>(1);
	double B13 = B.at<double>(2);
	double B23 = B.at<double>(3);
	double B33 = B.at<double>(4);

	double cx = -B13 / B11;
	double cy = -B23 / B22;
	double lmd = B33 - (B13 * B13 / B11 + B23 * B23 / B22);
	double fx = sqrt(lmd / B11);
	double fy = sqrt(lmd / B22);
	std::cout << "内参：" << "fx: " << fx << " fy: " << fy
		<< " cx: " << cx << " cy: " << cy
		<< " lmd: " << lmd << std::endl;

	K.at<double>(0, 0) = fx;
	K.at<double>(1, 1) = fy;
	K.at<double>(0, 2) = cx;
	K.at<double>(1, 2) = cy;
}

void estimateExtrinsics(const std::vector<cv::Mat>& homoVec, const cv::Mat& K, std::vector<cv::Mat>& rvec, std::vector<cv::Mat>& tvec)
{
	cv::Mat invK = K.inv();
	for (int i = 0; i < homoVec.size(); ++i)
	{
		cv::Mat rrt = invK * homoVec[i];
		double lmd = (norm(rrt.col(0)) + norm(rrt.col(1))) / 2;
		rrt = rrt / lmd;
		cv::Mat r1 = rrt.col(0).clone();
		cv::Mat r2 = rrt.col(1).clone();
		cv::Mat t = rrt.col(2).clone();
		cv::Mat r3 = r1.cross(r2);
		cv::Mat R = cv::Mat::zeros(cv::Size(3, 3), CV_64F);
		R.at<double>(0, 0) = r1.at<double>(0);
		R.at<double>(1, 0) = r1.at<double>(1);
		R.at<double>(2, 0) = r1.at<double>(2);
		R.at<double>(0, 1) = r2.at<double>(0);
		R.at<double>(1, 1) = r2.at<double>(1);
		R.at<double>(2, 1) = r2.at<double>(2);
		R.at<double>(0, 2) = r3.at<double>(0);
		R.at<double>(1, 2) = r3.at<double>(1);
		R.at<double>(2, 2) = r3.at<double>(2);

		cv::Mat U, W, VT;
		cv::SVD::compute(R, W, U, VT);
		R = U * VT;
		cv::Mat aa;
		Rodrigues(R, aa);
		rvec.push_back(aa);
		tvec.push_back(t);
#ifdef DEBUG
		std::cout << "r: " << aa.t() << std::endl;
		std::cout << "t: " << t.t() << std::endl;
#endif
	}
}

ceres::Solver::Options setCeresOptions()
{
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 50;
	options.gradient_tolerance = 1e-10;
	options.function_tolerance = 1e-10;
	options.parameter_tolerance = 1e-8;
	return options;
}

struct ReprojectionError
{
	ReprojectionError(double u, double v, double x, double y, double z) : u(u), v(v), x(x), y(y), z(z) {};
	template <typename T>
	bool operator()(const T* const camera, const T* const pose, T* residuals) const
	{
		// pose[0,1,2] 旋转
		// pose[3,4,5] 平移
		// camera[0,1] 焦距
		// camera[2,3] 主点
		// camera[4,5,6,7,8] 畸变k1、k2、k3、p1、p2
		T point[3] = { (T)x,(T)y,(T)z };
		T p[3];
		ceres::AngleAxisRotatePoint(pose, point, p);

		p[0] += pose[3];
		p[1] += pose[4];
		p[2] += pose[5];

		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		const T& k1 = camera[4];
		const T& k2 = camera[5];
		const T& k3 = camera[6];
		const T& p1 = camera[7];
		const T& p2 = camera[8];

		T r2 = xp * xp + yp * yp;
		T r4 = r2 * r2;
		T r6 = r2 * r4;
		T xd = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
		T yd = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp);

		const T& fx = camera[0];
		const T& fy = camera[1];
		const T& cx = camera[2];
		const T& cy = camera[3];

		T ud = fx * xd + cx;
		T vd = fy * yd + cy;

		residuals[0] = ud - (T)u;
		residuals[1] = vd - (T)v;

		return true;
	}

	//static ceres::CostFunction* create(const double u, const double v, const double x, const double y, const double z)
	//{
	//	return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 6>(new ReprojectionError(u, v, x, y, z)));
	//}
	double u, v, x, y, z;

};

void ceresCalibrate(const std::vector<std::vector<cv::Point2f>> imgPntsVec, std::vector<std::vector<cv::Point3f>>& objPntsVec, cv::Mat& K, cv::Mat& D, std::vector<cv::Mat>& rvec, std::vector<cv::Mat>& tvec)
{
		std::cout << "Ceres优化" << std::endl;
		double* x = new double[9]();
		double* p = new double[6 * imgPntsVec.size()]();
		x[0] = K.at<double>(0, 0);
		x[1] = K.at<double>(1, 1);
		x[2] = K.at<double>(0, 2);
		x[3] = K.at<double>(1, 2);
		x[4] = D.at<double>(0);
		x[5] = D.at<double>(1);
		x[6] = D.at<double>(3);
		x[7] = D.at<double>(4);
		x[8] = D.at<double>(2);

		for (size_t i = 0; i < imgPntsVec.size(); ++i)
		{
			p[6 * i + 0] = rvec[i].at<double>(0);
			p[6 * i + 1] = rvec[i].at<double>(1);
			p[6 * i + 2] = rvec[i].at<double>(2);
			p[6 * i + 3] = tvec[i].at<double>(0);
			p[6 * i + 4] = tvec[i].at<double>(1);
			p[6 * i + 5] = tvec[i].at<double>(2);
		}

		ceres::Problem problem;

		for (size_t i = 0; i < imgPntsVec.size(); ++i)
		{
			for (size_t j = 0; j < imgPntsVec[i].size(); ++j)
			{
				double uu = imgPntsVec[i][j].x;
				double vv = imgPntsVec[i][j].y;
				double xx = objPntsVec[i][j].x;
				double yy = objPntsVec[i][j].y;
				double zz = objPntsVec[i][j].z;
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 6>(new ReprojectionError(uu, vv, xx, yy, zz));
				problem.AddResidualBlock(cost_function, nullptr, x, p + 6 * i);
			}
		}
		ceres::Solver::Options options = setCeresOptions();

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";
		K.at<double>(0, 0) = x[0];
		K.at<double>(1, 1) = x[1];
		K.at<double>(0, 2) = x[2];
		K.at<double>(1, 2) = x[3];
		D.at<double>(0) = x[4];
		D.at<double>(1) = x[5];
		D.at<double>(2) = x[7];
		D.at<double>(3) = x[8];
		D.at<double>(4) = x[6];
		for (size_t i = 0; i < imgPntsVec.size(); ++i)
		{
			rvec[i].at<double>(0) = p[6 * i + 0];
			rvec[i].at<double>(1) = p[6 * i + 1];
			rvec[i].at<double>(2) = p[6 * i + 2];
			tvec[i].at<double>(0) = p[6 * i + 3];
			tvec[i].at<double>(1) = p[6 * i + 4];
			tvec[i].at<double>(2) = p[6 * i + 5];
		}
		delete[]p;
		delete[]x;
}

void displayResults(const cv::Mat& K, const cv::Mat& D, const std::vector<cv::Mat>& rvec, const std::vector<cv::Mat>& tvec)
{
	std::cout << "Intrinsics Matrix: " << std::endl << K << std::endl;
	std::cout << "Distortion: " << D << std::endl;
	for (size_t i = 0; i < rvec.size(); ++i)
	{
		std::cout << "Index: " << i << " Rotation: " << rvec[i].t() << " Translation: " << tvec[i].t() << std::endl;
	}
}

cv::Point2f project3DPoint(const cv::Point3f& p3, const cv::Mat& r, const cv::Mat& t, const cv::Mat& a, const cv::Mat& d)
{
	cv::Mat p = cv::Mat(3, 1, CV_64F);
	p.at<double>(0) = p3.x;
	p.at<double>(1) = p3.y;
	p.at<double>(2) = p3.z;
	cv::Mat R;
	cv::Rodrigues(r, R);
	p = R * p + t;
	double x = p.at<double>(0) / p.at<double>(2);
	double y = p.at<double>(1) / p.at<double>(2);
	double r2 = x * x + y * y;
	double r4 = r2 * r2;
	double r6 = r4 * r2;

	double k1 = d.at<double>(0);
	double k2 = d.at<double>(1);
	double p1 = d.at<double>(2);
	double p2 = d.at<double>(3);
	double k3 = d.at<double>(4);

	double xd = x * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
	double yd = y * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y);

	double fx = a.at<double>(0, 0);
	double fy = a.at<double>(1, 1);
	double cx = a.at<double>(0, 2);
	double cy = a.at<double>(1, 2);

	double u = fx * xd + cx;
	double v = fy * yd + cy;
	return cv::Point2f(u, v);
}

void calculateReprojectionError(const std::vector<std::vector<cv::Point2f>> imgPntsVec, const std::vector<std::vector<cv::Point3f>>& objPntsVec, const cv::Mat& K, const cv::Mat& D, const std::vector<cv::Mat>& rvec, const std::vector<cv::Mat>& tvec)
{
	double error = 0;
	size_t count = 0;
	for (size_t i = 0; i < imgPntsVec.size(); ++i)
	{
		for (size_t j = 0; j < imgPntsVec[i].size(); ++j)
		{
			cv::Point2f prjpnt = project3DPoint(objPntsVec[i][j], rvec[i], tvec[i], K, D);
			const cv::Point2f imgpnt = imgPntsVec[i][j];
			error += sqrt((imgpnt.x - prjpnt.x) * (imgpnt.x - prjpnt.x) + (imgpnt.y - prjpnt.y) * (imgpnt.y - prjpnt.y));
			++count;
		}
	}
	std::cout << "valid points number: " << count << std::endl;
	error /= count;
	std::cout << "reprojection error: " << error << std::endl;
}

void cvCalibrate(const std::vector<std::vector<cv::Point2f>> imgPntsVec, const std::vector<std::vector<cv::Point3f>>& objPntsVec, cv::Mat& K, cv::Mat& D, std::vector<cv::Mat>& rvec, std::vector<cv::Mat>& tvec)
{
	std::cout << "OpenCV标定" << std::endl;
	cv::calibrateCamera(objPntsVec, imgPntsVec, cv::Size(640,480), K, D, rvec, tvec);
}

int main()
{
	// 图片文件夹路径
	std::string path = "../data";
	// 查找的图片格式
	std::string pattern = "/*.png";
	// 存储图片的vector
	std::vector<cv::Mat> imageData;
	// 查找并读取图片
	readImages(path + pattern, imageData);

	// 棋盘格长、宽、尺度
	const uint width = 11;
	const uint height = 8;
	const double scale = 1.0;
	std::vector<std::vector<cv::Point3f>> objectPointsVector;
	std::vector<std::vector<cv::Point2f>> imagePointsVector;
	// 查找角点
	findCorners(imageData, width, height, scale, imagePointsVector, objectPointsVector);

	// 计算单应
	std::vector<cv::Mat> homographyVector;
	findAllHomography(imagePointsVector, objectPointsVector, homographyVector);

	// 估计内参
	cv::Mat K = cv::Mat::eye(3,3,CV_64F);
	estimateIntrinsics(homographyVector, K);

	// 估计外参
	std::vector<cv::Mat> rotationVector, translationVector;
	estimateExtrinsics(homographyVector, K, rotationVector, translationVector);

	// Ceres优化标定
	cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
	ceresCalibrate(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);
	displayResults(K, D, rotationVector, translationVector);
	calculateReprojectionError(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);

	// OpenCV标定
	cvCalibrate(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);
	displayResults(K, D, rotationVector, translationVector);
	calculateReprojectionError(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);
	return 0;
}