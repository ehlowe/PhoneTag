import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';

class ObjectDetectionPage extends StatefulWidget {
  @override
  _BasicOnnxInferencePageState createState() => _BasicOnnxInferencePageState();
}

class _BasicOnnxInferencePageState extends State<ObjectDetectionPage> {
  OrtSession? _onnxSession;
  File? _image;
  List<String>? _results;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _initializeOnnx();
  }

  Future<void> _initializeOnnx() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions()..setIntraOpNumThreads(1);
    const assetFileName = 'assets/yolov8n.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _onnxSession = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    setState(() {
      _loading = true;
      _image = File(image.path);
    });
    await _runInference();
  }

  Future<void> _runInference() async {
    if (_image == null || _onnxSession == null) return;

    // Load and preprocess the image
    final imageData = await _image!.readAsBytes();
    final decodedImage = img.decodeImage(imageData);
    if (decodedImage == null) return;

    final resizedImage = img.copyResize(decodedImage, width: 224, height: 224);
    final inputData = _imageToFloatList(resizedImage);

    // Create input tensor
    final shape = [1, 3, 224, 224];
    final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, shape);

    // Run inference
    final inputs = {'input': inputTensor};

    final runOptions = OrtRunOptions();
    final outputs = await _onnxSession!.run(runOptions, inputs);
    print(outputs);

    // // Process output
    // final outputTensor = (outputs.first as OrtValueTensor?)!;
    // final outputData = outputTensor.tensorData as Float32List;
    // final results = _processOutput(outputData);

    // setState(() {
    //   _results = results;
    //   _loading = false;
    // });

    // Clean up
    inputTensor.release();
    runOptions.release();
    outputs.forEach((value) => value?.release());
  }

  List<double> _imageToFloatList(img.Image image) {
    final floatList = List<double>.filled(3 * 224 * 224, 0);
    var idx = 0;
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        // Normalize pixel values to [0, 1]
        floatList[idx] = img.getRed(pixel) / 255.0;
        floatList[idx + 224 * 224] = img.getGreen(pixel) / 255.0;
        floatList[idx + 2 * 224 * 224] = img.getBlue(pixel) / 255.0;
        idx++;
      }
    }
    return floatList;
  }

  List<String> _processOutput(List<double> outputData) {
    // This is a simplified output processing.
    // You should replace this with actual label mapping for your model.
    final topK = 5;
    final List<MapEntry<int, double>> enumerated = outputData.asMap().entries.toList();
    enumerated.sort((a, b) => b.value.compareTo(a.value));
    return enumerated.take(topK).map((e) => 'Class ${e.key}: ${e.value.toStringAsFixed(4)}').toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Basic ONNX Inference'),
      ),
      body: Center(
        child: _loading
            ? CircularProgressIndicator()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  if (_image != null) ...[
                    Image.file(_image!, height: 200),
                    SizedBox(height: 20),
                  ],
                  ElevatedButton(
                    onPressed: _pickImage,
                    child: Text('Pick an image'),
                  ),
                  if (_results != null) ...[
                    SizedBox(height: 20),
                    Text('Top 5 predictions:'),
                    ...(_results!.map((result) => Text(result)).toList()),
                  ],
                ],
              ),
      ),
    );
  }

  @override
  void dispose() {
    _onnxSession?.release();
    OrtEnv.instance.release();
    super.dispose();
  }
}