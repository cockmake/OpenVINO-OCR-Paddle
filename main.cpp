#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
#include<openvino/openvino.hpp>


using namespace std;
using namespace cv;
using namespace ov;
using namespace chrono;


InferRequest inferRequest, inferRequest2;
Shape inputShape, inputShape2, outputShape2;
Tensor inputTensor, outputTensor, inputTensor2, outputTensor2;
float confidence = 0.5, confidence2 = 0.7;
size_t c, h, w, h2, w2;
size_t len, num;
float* output, *output2;
float width_ratio, height_ratio;
vector<string> dict;

Mat get_infer_mask(const Mat& src)
{
	//获取输入数据
	width_ratio = src.cols / static_cast<float>(w), height_ratio = src.rows / static_cast<float>(h);

	Mat blobImg;
	resize(src, blobImg, Size(w, h));

	blobImg.convertTo(blobImg, CV_32FC3);
	blobImg /= 255.0;

	vector<Mat> result;
	split(blobImg, result);
	result[0] = (result[0] - 0.485) / 0.229;
	result[1] = (result[1] - 0.456) / 0.224;
	result[2] = (result[2] - 0.406) / 0.225;
	merge(result, blobImg);



	float* input_data = inputTensor.data<float>();
	for (size_t i = 0; i < c; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			for (size_t k = 0; k < w; k++)
			{
				input_data[i * h * w + j * w + k] = blobImg.at<Vec3f>(j, k)[i];
			}
		}
	}
	
	inferRequest.infer();
	outputTensor = inferRequest.get_output_tensor(0);
	output = outputTensor.data<float>();

	Mat ret = Mat(h, w, CV_32FC1, output);
	threshold(ret, ret, confidence, 255, THRESH_BINARY);
	ret.convertTo(ret, CV_8UC1);

	return ret;
}

string get_text_ret(const Mat& src)
{
	//进行PaddleOCR的预处理
	//文字区的预处理需要做按比例放缩   该模型只支持  不加变向器为横向文本检测
	Mat blobImg = Mat::zeros(Size(320, 32), CV_8UC3), src_clone = src.clone();
	int src_clone_w = src_clone.cols, src_clone_h = src_clone.rows;
	float ratio = src_clone_h / 32.0;    //假设原本输入的文字就是正常文字(不瘦也不胖) 只是不符合检测的输入
	Size sz = Size(src_clone_w / ratio, 32);
	cv::resize(src_clone, src_clone, sz);
	if (src_clone.cols <= 320) {
		src_clone.copyTo(blobImg(Rect(0, 0, src_clone.cols, src_clone.rows)));
	}
	else {
		cv::resize(src_clone, blobImg, Size(320, 32));
	}

	blobImg.convertTo(blobImg, CV_32FC3);
	blobImg /= 255.0;
	vector<Mat> result;
	split(blobImg, result);
	result[0] = (result[0] - 0.485) / 0.229;
	result[1] = (result[1] - 0.456) / 0.224;
	result[2] = (result[2] - 0.406) / 0.225;
	merge(result, blobImg);
	//到此PaddleOCR的输入预处理完成



	float* input_data2 = inputTensor2.data<float>();

	for (size_t i = 0; i < c; i++)
	{
		for (size_t j = 0; j < h2; j++)
		{
			for (size_t k = 0; k < w2; k++)
			{
				input_data2[i * h2 * w2 + j * w2 + k] = blobImg.at<Vec3f>(j, k)[i];
			}
		}
	}

	inferRequest2.infer();
	outputTensor2 = inferRequest2.get_output_tensor(0);
	output2 = outputTensor2.data<float>();

	int idx;
	string ret = "";
	float* tmp;
	for (size_t i = 0; i < len; i++) {
		tmp = max_element(output2 + i * num, output2 + (i + 1) * num);
		if (*tmp > confidence2) {
			idx = tmp - (output2 + i * num);
			if (idx != 0) {
				ret += dict[idx];
			}
		}
	}
	return ret;
}
void init_model() {
	//初始化推理请求
	Core ie;
	String model_path = "D:\\GoogleDownload\\ch_PP-OCRv2_det_infer\\inference.pdmodel";
	shared_ptr<Model> p_model = ie.read_model(model_path);
	p_model->reshape({ {1, 3, 960, 960} });
	CompiledModel compiledModel = ie.compile_model(p_model, "CPU");
	inferRequest = compiledModel.create_infer_request();
	
	//获取输入输输出的Shape
	inputTensor = inferRequest.get_input_tensor(0);
	inputShape = inputTensor.get_shape();
	c = inputShape[1];
	h = inputShape[2];
	w = inputShape[3];
}
void init_model2() {

	Core ie;
	String model_path = "D:\\GoogleDownload\\ch_PP-OCRv2_rec_infer\\inference.pdmodel";
	shared_ptr<Model> p_model = ie.read_model(model_path);
	p_model->reshape({ {1, 3, 32, 320} });
	CompiledModel compiledModel = ie.compile_model(p_model, "CPU");
	inferRequest2 = compiledModel.create_infer_request();
	inputTensor2 = inferRequest2.get_input_tensor();
	inputShape2 = inputTensor2.get_shape();
	h2 = inputShape2[2];
	w2 = inputShape2[3];
	outputShape2 = inferRequest2.get_output_tensor().get_shape();
	len = outputShape2[1];
	num = outputShape2[2];
}
vector<Rect> process_mask(Mat& ret) {

	Mat kernel = getStructuringElement(MORPH_RECT, Size(35, 25));
	morphologyEx(ret, ret, MORPH_DILATE, kernel);
	vector<vector<Point>> contours;
	findContours(ret, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<Rect> boxs;
	RotatedRect rr;
	Rect box;
	for (int i = 0; i < contours.size(); i++) {
		rr = minAreaRect(contours[i]);
		box = rr.boundingRect();
		Rect new_box = Rect(box.x * width_ratio, box.y * height_ratio, box.width * width_ratio, box.height * height_ratio);
		boxs.emplace_back(new_box);
	}
	return boxs;
}
void init_dict() {
	//加载词典
	string dict_path = "C:\\Users\\make\\Desktop\\orc_dict.txt";
	dict.push_back("");
	fstream file;
	file.open(dict_path, ios::in);
	string tmp;
	if (file.is_open()) {
		while (!file.eof()) {
			file >> tmp;
			dict.emplace_back(tmp);
		}
		file.close();
	}
}
int main() {

	init_dict();
	init_model(); //加载检测模型
	init_model2(); //加载识别模型
	String img_path = "D:\\a.jpg";
	Mat img = imread(img_path), ret;
	time_point<steady_clock> start = steady_clock::now();
	ret = get_infer_mask(img);
	vector<Rect> boxs = process_mask(ret);
	for (int i = 0; i < boxs.size(); i++) {
		//可以加个判断 是否对该box覆盖的区域进行处理
		string a = get_text_ret(img(boxs[i]));
		rectangle(img, boxs[i], Scalar(0, 0, 255), 2);
		cout << a << endl;
	}
	imshow("1", img);
	waitKey(0);
	/*duration<double> dur = steady_clock::now() - start;*/
	return 0;
}