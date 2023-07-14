#include<iostream>
#include<opencv2/opencv.hpp>
#include<ceres/ceres.h>
#include<ceres/rotation.h>

bool readImages(const std::string& pattern, std::vector<cv::Mat>& images)
{
	// ����·��������pngͼƬ���ǵݹ顣
	std::vector<std::string> list;
	cv::glob(pattern, list);

	if (list.empty())
	{ 
		std::cout << "δ��·�����ҵ�PNG��ʽͼƬ!" << std::endl;
		return false;
	}
	else
	{
		std::cout << "�ҵ�ͼƬ������" << list.size() << std::endl;
		for (size_t i = 0; i < list.size(); ++i)
		{
			std::cout << list[i] << std::endl;
			// ��ȡͼƬ��vector<Mat>
			images.push_back(cv::imread(list[i], cv::IMREAD_UNCHANGED));
		}
		return true;
	}
}

bool findCorners(const std::vector<cv::Mat>& images, const uint& w, const uint& h, const double& scale, std::vector<std::vector<cv::Point2f>>& imgPntsVec, std::vector<std::vector<cv::Point3f>>& objPntsVec)
{
	std::vector<cv::Point2f> imgPoints;
	std::vector<cv::Point3f> objPoints;
	// ������������ϵ�еĽǵ�����vector
	for (size_t i = 0; i < h; ++i)
	{
		for (size_t j = 0; j < w; ++j)
		{
			objPoints.push_back(cv::Point3f(i * scale, j * scale, 0));
		}
	}

	size_t count = 0;
	for (uint i = 0; i < images.size(); ++i)
	{
		// ����OpenCV�������̸�ǵ㣬findChessboardCornersSB���ܹ�ֱ�ӻ�ȡ���̸�ǵ�����������ĺ�����
		bool found = cv::findChessboardCornersSB(images[i], cv::Size(w, h), imgPoints, 0);
		if (found)
		{
			imgPntsVec.push_back(imgPoints);
			objPntsVec.push_back(objPoints);
			count++;
			// ��ʾ���̸�����
			//cv::namedWindow("���ҵ��Ľǵ�");
			//cv::drawChessboardCorners(images[i], cv::Size(w, h), imgPoints, true);
			//cv::imshow("���ҵ��Ľǵ�",images[i]);
			//cv::waitKey();
			//cv::destroyWindow("���ҵ��Ľǵ�");
		}
	}

	if (count < 3)
	{
		std::cout << "��������Ч���̸�궨������С��3��" << std::endl;
		return false;
	}
	else
	{
		std::cout << "������Ч���̸�궨������Ϊ��" << count << std::endl;
		return true;
	}
}

cv::Mat findHomography(const std::vector<cv::Point2f>& imgPoints, const std::vector<cv::Point3f>& objPoints)
{
	cv::Mat homo;
	const size_t number = imgPoints.size();
	cv::Mat A(number * 2, 9, CV_64F);
	// ����A����������ⵥӦ
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
	// ���Ax=0
	cv::SVD::solveZ(A, homo);
	//std::cout << "��Ӧ����" << homo.t() << std::endl;
	// ��Ӧ����Ĵ����������һ��Ԫ��Ϊ1������Ϊ����
	homo = homo / homo.at<double>(8);
	// ��Ӧ����Ĵ����������һ��Ԫ��Ϊ������ζ�ű궨��ԭ�������ǰ����
	//if (homo.at<double>(8) < 0)
	//	homo = -homo;
	homo = homo.reshape(0, 3);
	//std::cout << homo << std::endl;
	return homo;
}

// �Ե���й�һ����ʹ�����ȶ���ʹ��ģ�������vector<Point2f>��vector<Point3f>
template<typename T>
void normalize(T& points, cv::Mat& N)
{
	double xmean = 0, ymean = 0, xstd = 0, ystd = 0;
	double num = (double)points.size();
	// �����ֵ
	for (size_t i = 0; i < points.size(); ++i)
	{
		xmean += points[i].x;
		ymean += points[i].y;
	}
	xmean /= num;
	ymean /= num;
	//std::cout << xmean << std::endl;
	//std::cout << ymean << std::endl;
	// �����׼��
	for (size_t i = 0; i < points.size(); ++i)
	{
		xstd += (points[i].x - xmean) * (points[i].x - xmean);
		ystd += (points[i].y - xmean) * (points[i].y - xmean);
	}
	xstd = cv::sqrt(xstd / num);
	ystd = cv::sqrt(ystd / num);
	//std::cout << xstd << std::endl;
	//std::cout << ystd << std::endl;
	// ���ƹ�һ��
	for (size_t i = 0; i < points.size(); ++i)
	{
		points[i].x = (points[i].x - xmean) / xstd * cv::sqrt(2);
		points[i].y = (points[i].y - ymean) / ystd * cv::sqrt(2);
	}
	N.at<double>(0, 0) = 1 / xstd * cv::sqrt(2);
	N.at<double>(0, 2) = -xmean / xstd * cv::sqrt(2);
	N.at<double>(1, 1) = 1 / ystd * cv::sqrt(2);
	N.at<double>(1, 2) = -ymean / ystd * cv::sqrt(2);
	//std::cout << N << std::endl;
}
//��ⵥӦ������һ��
cv::Mat findHomographyWithNormalization(const std::vector<cv::Point2f>& imgPoints, const std::vector<cv::Point3f>& objPoints)
{
	cv::Mat homo;
	const size_t number = imgPoints.size();
	cv::Mat A(number * 2, 9, CV_64F);
	// ����һ��
	std::vector<cv::Point3f> objNPnts{objPoints};
	cv::Mat No = cv::Mat::eye(3, 3, CV_64F);
	normalize(objNPnts, No);
	// ͼ����һ��
	std::vector<cv::Point2f> imgNPnts{imgPoints};
	cv::Mat Ni = cv::Mat::eye(3, 3, CV_64F);
	normalize(imgNPnts, Ni);

	// ����A����������ⵥӦ
	for (size_t i = 0; i < number; ++i)
	{
		double u = imgNPnts[i].x;
		double v = imgNPnts[i].y;
		double x = objNPnts[i].x;
		double y = objNPnts[i].y;

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
	// ���Ax=0
	cv::SVD::solveZ(A, homo);
	homo = homo.reshape(0, 3);
	homo = Ni.inv() * homo * No;
	// ��Ӧ����Ĵ����������һ��Ԫ��Ϊ1������Ϊ��ֵ��
	homo = homo / homo.at<double>(8);
	//std::cout << homo << std::endl;
	return homo;
}

void findAllHomography(const std::vector<std::vector<cv::Point2f>>& imgPntsVec, const std::vector<std::vector<cv::Point3f>>& objPntsVec, std::vector<cv::Mat>& homoVec)
{
	// �ҵ�����ͼ��ĵ�Ӧ����
	for (size_t i = 0; i < imgPntsVec.size(); ++i)
	{
		homoVec.push_back(findHomographyWithNormalization(imgPntsVec[i], objPntsVec[i]));
		//homoVec.push_back(findHomography(imgPntsVec[i], objPntsVec[i]));
	}
}

void estimateIntrinsics(const std::vector<cv::Mat>& homoVec, cv::Mat K)
{
	cv::Mat v(homoVec.size() * 2, 5, CV_64F);
	// ������Ԫ�ظ�ֵ
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
	// ���B����ͬ����Ax=0����
	cv::Mat B;
	cv::SVD::solveZ(v, B);
	std::cout << "B:" << B.t() << std::endl;

	double B11 = B.at<double>(0);
	double B22 = B.at<double>(1);
	double B13 = B.at<double>(2);
	double B23 = B.at<double>(3);
	double B33 = B.at<double>(4);
	// ��B����ָ��ڲΣ��ٶ�skewnessΪ��
	double cx = -B13 / B11;
	double cy = -B23 / B22;
	double lmd = B33 - (B13 * B13 / B11 + B23 * B23 / B22);
	double fx = sqrt(lmd / B11);
	double fy = sqrt(lmd / B22);
	std::cout << "�ڲΣ�" << "fx: " << fx << " fy: " << fy
		<< " cx: " << cx << " cy: " << cy
		<< " lmd: " << lmd << std::endl;
	// ���ڲξ���ֵ
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

		// SVD�ֽ⣬�ҵ�����Լ������ת����
		cv::Mat U, W, VT;
		cv::SVD::compute(R, W, U, VT);
		R = U * VT;
		// ��ת����ת���
		cv::Mat aa;
		Rodrigues(R, aa);
		rvec.push_back(aa);
		tvec.push_back(t);
		//std::cout << "r: " << aa.t() << std::endl;
		//std::cout << "t: " << t.t() << std::endl;

	}
}

ceres::Solver::Options setCeresOptions()
{
	// Ceres�Ż�����������
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 50; // CeresĬ��ֵ
	options.gradient_tolerance = 1e-10; // CeresĬ��ֵ
	options.function_tolerance = 1e-10; // ��Ĭ��ֵ1e-6������1e-10�����Զ�������Σ��ﵽ��С����ͶӰ���
	options.parameter_tolerance = 1e-8; // CeresĬ��ֵ
	return options;
}

struct ReprojectionError
{
	// Ceres�Ż�����Ľṹ�壬������ͶӰ���
	// ���캯��
	ReprojectionError(double u, double v, double x, double y, double z) : u(u), v(v), x(x), y(y), z(z) {};
	template <typename T>
	bool operator()(const T* const camera, const T* const pose, T* residuals) const
	{
		// pose[0,1,2] ��ת
		// pose[3,4,5] ƽ��
		// camera[0,1] ����
		// camera[2,3] ����
		// camera[4,5,6,7,8] ����k1��k2��k3��p1��p2
		T point[3] = { (T)x,(T)y,(T)z };
		T p[3];
		// Ӧ����ת������궨������ϵ�еĵ�
		ceres::AngleAxisRotatePoint(pose, point, p);
		// ����ƽ�ƾ��󣬳�Ϊ�������ϵ�еĵ�
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

		// ��ӻ���
		T r2 = xp * xp + yp * yp;
		T r4 = r2 * r2;
		T r6 = r2 * r4;
		T xd = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
		T yd = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp);

		const T& fx = camera[0];
		const T& fy = camera[1];
		const T& cx = camera[2];
		const T& cy = camera[3];

		// ת��Ϊ��������
		T ud = fx * xd + cx;
		T vd = fy * yd + cy;

		// ��ͶӰ���
		residuals[0] = ud - (T)u;
		residuals[1] = vd - (T)v;

		return true;
	}

	double u, v, x, y, z;
};

void ceresCalibrate(const std::vector<std::vector<cv::Point2f>> imgPntsVec, std::vector<std::vector<cv::Point3f>>& objPntsVec, cv::Mat& K, cv::Mat& D, std::vector<cv::Mat>& rvec, std::vector<cv::Mat>& tvec)
{
		std::cout << "Ceres�Ż�" << std::endl;
		// ����cam�洢����ڲ�
		// cam[0,1] ����
		// cam[2,3] ����
		// cam[4,5,6,7,8] ����k1��k2��k3��p1��p2
		double* cam = new double[9]();
		// pose[0,1,2] ��ת
		// pose[3,4,5] ƽ��
		double* pose = new double[6 * imgPntsVec.size()]();
		// �����ֵ��KΪ���������ѱ궨�������ڲΣ�DΪ����
		cam[0] = K.at<double>(0, 0);
		cam[1] = K.at<double>(1, 1);
		cam[2] = K.at<double>(0, 2);
		cam[3] = K.at<double>(1, 2);
		// ע��Opencv��D�Ļ���˳��Ϊk1��k2��p1��p2��k3��Ϊ�˷����OpenCV�Ƚϣ���Ҫ����һ��
		cam[4] = D.at<double>(0);
		cam[5] = D.at<double>(1);
		cam[6] = D.at<double>(3);
		cam[7] = D.at<double>(4);
		cam[8] = D.at<double>(2);
		// �����ֵ��rvec��tvecΪ�����ѱ궨���������
		for (size_t i = 0; i < imgPntsVec.size(); ++i)
		{
			pose[6 * i + 0] = rvec[i].at<double>(0);
			pose[6 * i + 1] = rvec[i].at<double>(1);
			pose[6 * i + 2] = rvec[i].at<double>(2);
			pose[6 * i + 3] = tvec[i].at<double>(0);
			pose[6 * i + 4] = tvec[i].at<double>(1);
			pose[6 * i + 5] = tvec[i].at<double>(2);
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
				// ��Ӳв��
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 6>(new ReprojectionError(uu, vv, xx, yy, zz));
				problem.AddResidualBlock(cost_function, nullptr, cam, pose + 6 * i);
			}
		}
		// �����Ż�ѡ��
		ceres::Solver::Options options = setCeresOptions();
		ceres::Solver::Summary summary;
		// ����Ceres����Ż�����
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";

		// �Ż������ֵ��K��D��rvec��tvec
		K.at<double>(0, 0) = cam[0];
		K.at<double>(1, 1) = cam[1];
		K.at<double>(0, 2) = cam[2];
		K.at<double>(1, 2) = cam[3];
		D.at<double>(0) = cam[4];
		D.at<double>(1) = cam[5];
		D.at<double>(2) = cam[7];
		D.at<double>(3) = cam[8];
		D.at<double>(4) = cam[6];
		for (size_t i = 0; i < imgPntsVec.size(); ++i)
		{
			rvec[i].at<double>(0) = pose[6 * i + 0];
			rvec[i].at<double>(1) = pose[6 * i + 1];
			rvec[i].at<double>(2) = pose[6 * i + 2];
			tvec[i].at<double>(0) = pose[6 * i + 3];
			tvec[i].at<double>(1) = pose[6 * i + 4];
			tvec[i].at<double>(2) = pose[6 * i + 5];
		}
		delete[] pose;
		delete[] cam;
}

void displayResults(const cv::Mat& K, const cv::Mat& D, const std::vector<cv::Mat>& rvec, const std::vector<cv::Mat>& tvec)
{
	// ��ӡ�Ż����
	std::cout << "�ڲξ���: " << std::endl << K << std::endl;
	std::cout << "�������: " << D << std::endl;
	for (size_t i = 0; i < rvec.size(); ++i)
	{
		std::cout << "ͼ������: " << i << "����ת����: " << rvec[i].t() << "��ƽ�ƾ���: " << tvec[i].t() << std::endl;
	}
}

cv::Point2f project3DPoint(const cv::Point3f& p3, const cv::Mat& r, const cv::Mat& t, const cv::Mat& a, const cv::Mat& d)
{
	// �������ģ�ͣ���3D��ͶӰ��2D��������
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
	// ������ͶӰ���
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
	std::cout << "��Ч�궨������: " << count << std::endl;
	error /= count;
	std::cout << "��ͶӰ���: " << error << std::endl;
}

int main()
{
	// ͼƬ�ļ���·��
	std::string path = "../data";
	// ���ҵ�ͼƬ��ʽ
	std::string pattern = "/*.png";
	// �洢ͼƬ��vector
	std::vector<cv::Mat> imageData;
	// ���Ҳ���ȡͼƬ
	readImages(path + pattern, imageData);

	// ���̸񳤡����߶�
	const uint width = 11;
	const uint height = 8;
	const double scale = 1.0;
	std::vector<std::vector<cv::Point3f>> objectPointsVector;
	std::vector<std::vector<cv::Point2f>> imagePointsVector;
	// ���ҽǵ�
	findCorners(imageData, width, height, scale, imagePointsVector, objectPointsVector);

	// ���㵥Ӧ
	std::vector<cv::Mat> homographyVector;
	findAllHomography(imagePointsVector, objectPointsVector, homographyVector);

	// �����ڲ�
	cv::Mat K = cv::Mat::eye(3,3,CV_64F);
	estimateIntrinsics(homographyVector, K);

	// �������
	std::vector<cv::Mat> rotationVector, translationVector;
	estimateExtrinsics(homographyVector, K, rotationVector, translationVector);

	// �����ֵ��Ϊ0
	cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
	// Ceres�Ż��궨
	ceresCalibrate(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);
	displayResults(K, D, rotationVector, translationVector);
	calculateReprojectionError(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);

	// OpenCV�궨
	std::cout << "OpenCV�궨" << std::endl;
	cv::calibrateCamera(objectPointsVector, imagePointsVector, imageData[0].size(), K, D, rotationVector, translationVector);
	displayResults(K, D, rotationVector, translationVector);
	calculateReprojectionError(imagePointsVector, objectPointsVector, K, D, rotationVector, translationVector);
	return 0;
}