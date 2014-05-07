#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/flann.hpp>

#include <nnforge/nnforge.h>
#include <nnforge/cuda/cuda.h>
#include <nnforge/snapshot_visualizer.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef std::vector< nnforge_shared_ptr<const std::vector<float> > > DataList;
typedef nnforge_shared_ptr<cv::flann::Index> FLANNIndexPtr;

const int kNumOutputMaps = 16;
const char *kDictAnnotationsPath = "./dict_annotations.bin";
const char *KDictProcessedPatchesPath = "./dict_processed_patches.bin";
const char *kDictPath = "./dict.bin";
const char *kWeightsPath = "./weights.bin";
const char *kKnnIndexPath = "./knn_index.bin";

template<unsigned int index_id> class VectorElementExtractor {
public:
	VectorElementExtractor() {
	}

	inline float operator()(cv::Vec3f x) const {
		return x[index_id];
	}
};

nnforge::network_schema_smart_ptr GetSchema() {
	nnforge::network_schema_smart_ptr schema(new nnforge::network_schema());

	// Input patch size is 34x34.

	// conv1.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 7), 3, 96))); // 28x28
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::rectified_linear_layer()));

	// pool1.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2, 2)))); // 14x14

	// conv2.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 5), 96, 128))); // 10x10
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::rectified_linear_layer()));

	// pool2.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2, 2)))); // 5x5

	// conv3.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 3), 128, 256))); // 3x3
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::rectified_linear_layer()));

	// fc4.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 3), 256, 768))); // 1x1
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::rectified_linear_layer()));

	// fc5.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), 768, 768))); // 1x1
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::rectified_linear_layer()));

	// fc6.
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), 768, kNumOutputMaps))); // 1x1

	return schema;
}

cv::Mat3f GetImage(const char *filename, float scale) {
	cv::Mat img = cv::imread(filename);

	// Convert image to the floating-point format.
	img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

	// Remove excessive pixel from the border. 321 x 481 -> 320 x 480 :).
	img = img(cv::Range(0, img.rows - 1), cv::Range(0, img.cols - 1)).clone();

	// Subtract mean.
	img -= cv::Scalar(0.37162688, 0.44378472, 0.43420864);

	// Scale image.
	cv::resize(img, img, cv::Size(), scale, scale);

	// Pad image.
	cv::copyMakeBorder(img, img, 15, 15 + 4, 15, 15 + 4, cv::BORDER_REFLECT);

	return img;
}

void ConvertToInputFormat(cv::Mat3f image, std::vector<float> &input_data) {
	input_data.resize(image.rows * image.cols * 3);

	// Red.
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin(),
		VectorElementExtractor<2U>());
	// Green.
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin() + (image.rows * image.cols),
		VectorElementExtractor<1U>());
	// Blue.
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin() + (image.rows * image.cols * 2),
		VectorElementExtractor<0U>());
}

void CreateShiftedInputs(const cv::Mat3f img, DataList &input_data_list) {
	input_data_list.resize(16);

	cv::Size img_size = img.size();

	int input_idx = 0;
	for (int y_shift = 0; y_shift < 4; ++y_shift) {
		for (int x_shift = 0; x_shift < 4; ++x_shift, ++input_idx) {
			cv::Mat3f roi =
				img(cv::Range(y_shift, y_shift + img_size.height - 4),
				    cv::Range(x_shift, x_shift + img_size.width - 4));

			std::vector<float> *input = new std::vector<float>();
			ConvertToInputFormat(roi, *input);
			input_data_list[input_idx].reset(input);
		}
	}
}

void AssembleMaps(
		const std::vector<std::vector<float> > &parts,
		std::vector<cv::Mat1f> &maps,
		cv::Size map_size,
		int maps_count) {

	const int stride = 4;

	maps.resize(maps_count);

	int part_height = map_size.height / 4;
	int part_width = map_size.width / 4;
	int map_stride = part_height * part_width;

	for (int map_idx = 0; map_idx < maps_count; ++map_idx) {
		cv::Mat1f map(map_size);

		for (int y_shift = 0; y_shift < 4; ++y_shift) {
			for (int x_shift = 0; x_shift < 4; ++x_shift) {
				const std::vector<float> &part = parts[4 * y_shift + x_shift];

				for (int y_out = y_shift, y_in = 0; y_out < map_size.height; y_out += stride, ++y_in) {
					for (int x_out = x_shift, x_in = 0; x_out < map_size.width; x_out += stride, ++x_in) {
						map.at<float>(y_out, x_out) =
							part[map_idx * map_stride +  y_in * part_width + x_in];
					}
				}
			}
		}

		maps[map_idx] = map;
	}
}

cv::Mat1f CombinePatches(
		const cv::Mat1i indices,
		const std::vector<cv::Mat1f> &patches,
		cv::Size output_size) {

	int patch_size = patches[0].rows;

	cv::Mat1f output = cv::Mat1f::zeros(output_size.height + patch_size - 1,
										output_size.width + patch_size - 1);
	cv::Mat1f counts = cv::Mat1f::zeros(output_size.height + patch_size - 1,
										output_size.width + patch_size - 1);

	int pixel_idx = 0;
	for (int y = 0; y < output_size.height; ++y) {
		for (int x = 0; x < output_size.width; ++x, ++pixel_idx) {
			cv::Mat1f output_roi = output(
				cv::Range(y, y + patch_size),
				cv::Range(x, x + patch_size));

			cv::Mat1f counts_roi = counts(
				cv::Range(y, y + patch_size),
				cv::Range(x, x + patch_size));

			const cv::Mat1f patch = patches[indices.at<int>(pixel_idx)];

			output_roi += patch;
			counts_roi += 1.0f;
		}
	}

	counts = cv::max(counts, 1.0f);
	output /= counts;

	output = output(cv::Range(8, 8 + output_size.height),
					cv::Range(8, 8 + output_size.width)).clone();

	return output;
}

void LoadDictAnnotations(std::vector<cv::Mat1f> &annotations) {
	const char *filename = kDictAnnotationsPath;

	std::ifstream f(filename, std::istream::binary);

	unsigned int num_annotations;
	f.read(reinterpret_cast<char *>(&num_annotations), sizeof(num_annotations));

	annotations.resize(num_annotations);

	for (int annotation_idx = 0; annotation_idx < num_annotations; ++annotation_idx) {
		std::vector<float> annotation_data(16 * 16);

		f.read(reinterpret_cast<char *> (&annotation_data[0]),
			   sizeof(float) * annotation_data.size());

		annotations[annotation_idx] = cv::Mat(16, 16, CV_32FC1, &annotation_data[0]).clone();
	}

	f.close();
}

void LoadDictPatches(DataList &input_data_list) {
	const char *filename = KDictProcessedPatchesPath;

	std::ifstream f(filename, std::istream::binary);

	unsigned int num_patches;
	f.read(reinterpret_cast<char *>(&num_patches), sizeof(num_patches));

	input_data_list.resize(num_patches);

	for (int patch_idx = 0; patch_idx < num_patches; ++patch_idx) {
		std::vector<float> &patch_data = *(new std::vector<float>(34 * 34 * 3));

		f.read(reinterpret_cast<char *> (&patch_data[0]),
			   sizeof(float) * patch_data.size());

		input_data_list[patch_idx].reset(&patch_data);
	}

	f.close();
}

cv::Mat1f GetFeaturesMat(const std::vector<std::vector<float> > &features) {
	int features_dim = features[0].size();
	int num_samples = features.size();

	cv::Mat1f output(cv::Size(features_dim, num_samples));

	for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
		const cv::Mat sample_features(
			cv::Size(features_dim, 1),
			CV_32FC1,
			const_cast<float *> (&(features[sample_idx][0])));

		sample_features.copyTo(output.row(sample_idx));
	}

	return output;
}

nnforge::network_tester_smart_ptr SetupTester(int device_id) {
	nnforge::network_tester_smart_ptr tester;
	nnforge::network_schema_smart_ptr schema(GetSchema());

	nnforge::cuda::factory_generator_cuda fg;
	fg.set_device_id(device_id);
	fg.set_max_global_memory_usage_ratio(0.8f);
	fg.initialize();
	tester = fg.create_tester_factory()->create(schema);

	nnforge::network_data_smart_ptr data(new nnforge::network_data());
	{
		boost::filesystem::ifstream in(kWeightsPath,
									   std::ios_base::in |
									   std::ios_base::binary);
		data->read(in);
	}

	tester->set_data(data);

	return tester;
}

void MakeLayerConfigurationForDict(
		nnforge::layer_configuration_specific &input_configuration,
		nnforge::layer_configuration_specific &output_configuration) {

	input_configuration.feature_map_count = 3;
	input_configuration.dimension_sizes.resize(2);
	input_configuration.dimension_sizes[0] = 34;
	input_configuration.dimension_sizes[1] = 34;

	output_configuration.feature_map_count = kNumOutputMaps;
	output_configuration.dimension_sizes.resize(2);
	output_configuration.dimension_sizes[0] = 1;
	output_configuration.dimension_sizes[1] = 1;
}

void MakeLayerConfigurationForImage(
		const cv::Mat3f img,
		nnforge::layer_configuration_specific &input_configuration,
		nnforge::layer_configuration_specific &output_configuration) {

	input_configuration.feature_map_count = 3;
	input_configuration.dimension_sizes.resize(2);
	input_configuration.dimension_sizes[0] = img.cols - 4;
	input_configuration.dimension_sizes[1] = img.rows - 4;

	output_configuration.feature_map_count = kNumOutputMaps;
	output_configuration.dimension_sizes.resize(2);
	output_configuration.dimension_sizes[0] = (img.cols - 4 - 30) / 4;
	output_configuration.dimension_sizes[1] = (img.rows - 4 - 30) / 4;
}

void DumpMatrix(const cv::Mat1f mat, const char *filename) {
	if (!mat.isContinuous()) {
		throw std::runtime_error("Can't dump a non-continuous matrix!");
	}

	std::ofstream f(filename, std::ofstream::binary);

	unsigned int rows = mat.rows;
	unsigned int cols = mat.cols;

	f.write(reinterpret_cast<char *> (&rows), sizeof(rows));
	f.write(reinterpret_cast<char *> (&cols), sizeof(cols));

	f.write(reinterpret_cast<char *> (mat.data), sizeof(float) * rows * cols);

	f.close();
}

cv::Mat1f ReadMatrix(const char *filename) {
	std::ifstream f(filename, std::ifstream::binary);

	unsigned int rows;
	unsigned int cols;

	f.read(reinterpret_cast<char *> (&rows), sizeof(rows));
	f.read(reinterpret_cast<char *> (&cols), sizeof(cols));

	cv::Mat1f mat(cv::Size(cols, rows));

	f.read(reinterpret_cast<char *> (mat.data), sizeof(float) * rows * cols);

	f.close();

	return mat;
}

void DumpFilters(
		nnforge::network_schema_smart_ptr schema,
		nnforge::network_data_smart_ptr data) {

	std::vector<nnforge::layer_data_configuration_list> layer_data_configuration_list_list =
		schema->get_layer_data_configuration_list_list();

	nnforge::snapshot_visualizer::save_ann_snapshot(*data, layer_data_configuration_list_list, "filters.png");
}

void GetImagesPaths(const std::string &root, std::vector<std::string> &images_paths) {
	if (!fs::exists(root)) {
		return;
	}

	if (fs::is_directory(root)) {
		fs::directory_iterator it(root);
		fs::directory_iterator endit;
		while (it != endit) {
			if (fs::is_regular_file(*it) &&
				it->path().extension().compare(".jpg") == 0) {

				images_paths.push_back(fs::absolute(it->path()).string());
			}
			++it;
		}
	} else if (fs::path(root).extension().compare(".txt") == 0) {
		std::ifstream f(root.c_str());
		std::string image_path;

		while (f.good()) {
			getline(f, image_path);
			if (image_path.empty()) {
				continue;
			}
			images_paths.push_back(fs::absolute(image_path).string());
		}

		f.close();
	} else {
		images_paths.push_back(root);
	}
}

int main(int argc, char* argv[]) {
	try {
		boost::chrono::steady_clock::time_point start;
		boost::chrono::duration<float> elapsed_time;
		boost::chrono::duration<float> test_time;

		//
		// Handle command-line parameters.
		//
		po::options_description desc("Allowed options");
		desc.add_options()
			("device-id,d", po::value<int>()->default_value(0), "device index")
			("process-dict-only", "process dictionary only")
		    ("source-path,s", po::value<std::string>(), "path to input image/directory/list")
			("target-path,t", po::value<std::string>(), "output directory")
			("scale", po::value<float>()->default_value(1.0f), "image scale");

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		std::cout << "=" << std::endl;

		int device_id = vm["device-id"].as<int>();
		std::cout << "= Device ID: " << device_id << std::endl;

		bool process_dict_only = vm.count("process-dict-only");

		std::string source_path;
		if (vm.count("source-path")) {
			source_path = vm["source-path"].as<std::string>();
		} else {
			source_path = "/home/yganin/Arbeit/Projects/NN/Segmentation/BSDS500/BSR/BSDS500/data/images/test/3063.jpg";
		}
		std::cout << "= Source path: " << source_path << std::endl;

		std::vector<std::string> images_paths;
		GetImagesPaths(source_path, images_paths);

		std::string target_path;
		if (vm.count("target-path")) {
			target_path = vm["target-path"].as<std::string>();
		} else {
			target_path = ".";
		}
		std::cout << "= Target path: " << target_path << std::endl;

		float image_scale = vm["scale"].as<float>();
		std::cout << "= Image scale: " << image_scale << std::endl;

		std::cout << "=" << std::endl;

		test_time = boost::chrono::duration<float>::zero();

		nnforge::cuda::cuda::init();

		//
		// Perform framework setup.
		//
		std::cout << "[*] Setting up framework..." << std::endl;
		start = boost::chrono::high_resolution_clock::now();

		nnforge::layer_configuration_specific input_configuration;
		nnforge::layer_configuration_specific output_configuration;
		nnforge::network_tester_smart_ptr tester = SetupTester(device_id);

		elapsed_time = boost::chrono::high_resolution_clock::now() - start;
		std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

		//
		// Load dictionary annotations, obtain neural codes for
		// dictionary patches and build a FLANN forest for a
		// subsequent search.
		//
		std::cout << "[*] Preparing dictionary..." << std::endl;
		start = boost::chrono::high_resolution_clock::now();

		std::vector<cv::Mat1f> annotations;
		LoadDictAnnotations(annotations);

		cv::Mat1f dict;
		if (fs::exists(kDictPath)) {
			dict = ReadMatrix(kDictPath);
		} else {
			MakeLayerConfigurationForDict(input_configuration, output_configuration);

			DataList input_data_list;
			LoadDictPatches(input_data_list);
			DataList output_data_list;

			nnforge::supervised_data_mem_reader reader(
				input_configuration,
				output_configuration,
				input_data_list,
				output_data_list);

			tester->set_input_configuration_specific(input_configuration);
			nnforge::output_neuron_value_set_smart_ptr output_ptr = tester->run(
				reader, 1);

			dict = GetFeaturesMat(output_ptr->neuron_value_list);

			DumpMatrix(dict, kDictPath);
		}

		FLANNIndexPtr knn_index;
		if (fs::exists(kKnnIndexPath)) {
			knn_index.reset(new cv::flann::Index(dict, cv::flann::SavedIndexParams(kKnnIndexPath)));
		} else {
			knn_index.reset(new cv::flann::Index(dict, cv::flann::KDTreeIndexParams(4)));
			knn_index->save(kKnnIndexPath);
		}

		elapsed_time = boost::chrono::high_resolution_clock::now() - start;
		std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

		if (process_dict_only) {
			return 0;
		}

		for (int img_idx = 0; img_idx < images_paths.size(); ++img_idx) {
			std::cout << "=== Processing image: " << images_paths[img_idx] << std::endl;
			//
			// Prepare inputs for the CNN.
			// Here we stack all possible shifts of the input image into a
			// single batch.
			//
			std::cout << "[*] Creating inputs..." << std::endl;

			cv::Mat3f img = GetImage(images_paths[img_idx].c_str(), image_scale);
			MakeLayerConfigurationForImage(img, input_configuration, output_configuration);

			start = boost::chrono::high_resolution_clock::now();

			DataList input_data_list;
			CreateShiftedInputs(img, input_data_list);
			DataList output_data_list;

			nnforge::supervised_data_mem_reader reader(
				input_configuration,
				output_configuration,
				input_data_list,
				output_data_list);

			elapsed_time = boost::chrono::high_resolution_clock::now() - start;
			test_time += elapsed_time;
			std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

			//
			// Get CNN's top layer activities.
			//
			std::cout << "[*] Performing forward pass..." << std::endl;

			tester->set_input_configuration_specific(input_configuration);

			start = boost::chrono::high_resolution_clock::now();

			nnforge::output_neuron_value_set_smart_ptr output_ptr = tester->run(
				reader, 1);

			elapsed_time = boost::chrono::high_resolution_clock::now() - start;
			test_time += elapsed_time;
			std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

			//
			// Rearrange obtained activities into output feature maps.
			//
			std::cout << "[*] Assembling output..." << std::endl;
			start = boost::chrono::high_resolution_clock::now();

			std::vector<cv::Mat1f> maps;
			AssembleMaps(output_ptr->neuron_value_list,
						 maps,
						 cv::Size(img.cols - 4 - 30, img.rows - 4 - 30),
						 kNumOutputMaps);

			elapsed_time = boost::chrono::high_resolution_clock::now() - start;
			test_time += elapsed_time;
			std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

			//
			// Match features against dictionary codes.
			//
			std::cout << "[*] Running kNN search..." << std::endl;
			start = boost::chrono::high_resolution_clock::now();

			int total_pixels = maps[0].total();

			cv::Mat1f dists(cv::Size(1, total_pixels));
			cv::Mat1i indices(cv::Size(1, total_pixels));
			{
				// Convert maps into queries for FLANN.
				cv::Mat1f queries(cv::Size(maps.size(), total_pixels));
				for (int map_idx = 0; map_idx < maps.size(); ++map_idx) {
					maps[map_idx].reshape(0, total_pixels).copyTo(queries.col(map_idx));
				}

				// Run search.
				knn_index->knnSearch(queries, indices, dists, 1, cv::flann::SearchParams(32));
			}

			elapsed_time = boost::chrono::high_resolution_clock::now() - start;
			test_time += elapsed_time;
			std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

			//
			// Finally, combine appropriate annotation patches into the output
			// edge map.
			//
			std::cout << "[*] Combining annotation patches..." << std::endl;
			start = boost::chrono::high_resolution_clock::now();

			cv::Mat1f edges_map = CombinePatches(indices, annotations, maps[0].size());

			elapsed_time = boost::chrono::high_resolution_clock::now() - start;
			test_time += elapsed_time;
			std::cout << "    Done in " << elapsed_time.count() << "s" << std::endl;

			std::string img_name =
				fs::path(images_paths[img_idx]).stem().string() + ".bin";
			DumpMatrix(edges_map, (fs::path(target_path) / img_name).c_str());
//			cv::imwrite((fs::path(target_path) / img_name).c_str(), edges_map * 255.0);
		}

		std::cout << "=" << std::endl;
		std::cout << "= Overall testing time is "
				  << test_time.count() << "s"
				  << std::endl;
		std::cout << "=" << std::endl;
	} catch (const std::exception& e) {
		std::cout << "[ERROR] Exception caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
