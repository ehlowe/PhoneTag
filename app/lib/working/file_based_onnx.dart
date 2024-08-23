import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';

class ObjectDetectionPage extends StatefulWidget {
  @override
  _ObjectDetectionPageState createState() => _ObjectDetectionPageState();
}

class _ObjectDetectionPageState extends State<ObjectDetectionPage> {
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

    final resizedImage = img.copyResize(decodedImage, width: 640, height: 640);
    final inputData = _imageToFloat32List(resizedImage);

    // Create input tensor
    final shape = [1, 3, 640, 640];
    final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, shape);

    // Run inference
    final inputs = {'images': inputTensor};

    final runOptions = OrtRunOptions();
    final outputs = await _onnxSession!.run(runOptions, inputs);
    
    final outputData = outputs[0]!.value as List<List<List<double>>>;
    final results = _processOutput(outputData);
    setState(() {
      _results = results;
      _loading = false;
    });

    // Clean up
    inputTensor.release();
    runOptions.release();
    outputs.forEach((value) => value?.release());
  }

  Float32List _imageToFloat32List(img.Image image) {
    final float32List = Float32List(3 * 640 * 640);
    var idx = 0;
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        // Normalize pixel values to [0, 1]
        float32List[idx] = img.getRed(pixel) / 255.0;
        float32List[idx + 640 * 640] = img.getGreen(pixel) / 255.0;
        float32List[idx + 2 * 640 * 640] = img.getBlue(pixel) / 255.0;
        idx++;
      }
    }
    return float32List;
  }

  List<String> _processOutput(List<List<List<double>>> outputData) {
    final results = <String>[];
    
    // outputData shape is [1, 84, 8400]
    final detections = outputData[0]; // [84, 8400]
    
    for (int i = 0; i < 8400; i++) {
      final detection = <double>[];
      for (int j = 0; j < 84; j++) {
        detection.add(detections[j][i]);
      }
      
      final x = detection[0];
      final y = detection[1];
      final w = detection[2];
      final h = detection[3];
      
      final classScores = detection.sublist(4);
      final maxScore = classScores.reduce((a, b) => a > b ? a : b);
      final classIndex = classScores.indexOf(maxScore);
      
      if (maxScore > 0.5) { // Confidence threshold
        results.add('Detection $i: (x: $x, y: $y, w: $w, h: $h), Class: $classIndex, Score: ${maxScore.toStringAsFixed(3)}');
      }
    }
    
    return results;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLOv8 Object Detection - Debug'),
      ),
      body: SingleChildScrollView(
        child: Center(
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
                      Text('Detected objects:'),
                      ..._results!.map((result) => Text(result, style: TextStyle(fontSize: 10))).toList(),
                    ],
                  ],
                ),
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