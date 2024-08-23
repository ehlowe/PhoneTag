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
  Map<String, dynamic>? _highestConfidenceDetection;
  bool _loading = false;
  Size? _imageSize;

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

    setState(() {
      _imageSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
    });

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
    final highestConfidenceDetection = _processOutput(outputData);
    print(highestConfidenceDetection);
    
    setState(() {
      _highestConfidenceDetection = highestConfidenceDetection;
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

  Map<String, dynamic>? _processOutput(List<List<List<double>>> outputData) {
    // outputData shape is [1, 84, 8400]
    final detections = outputData[0]; // [84, 8400]
    
    Map<String, dynamic>? highestConfidenceDetection;
    double highestConfidence = 0;

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
      
      if (maxScore > highestConfidence){
        if  (classIndex == 0) {
          highestConfidence = maxScore;
          highestConfidenceDetection = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'class': classIndex,
            'score': maxScore,
          };
        }
      }
    }
    
    return highestConfidenceDetection;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLOv8 Object Detection - Bounding Box'),
      ),
      body: Center(
        child: _loading
            ? CircularProgressIndicator()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  if (_image != null && _imageSize != null) ...[
                    Stack(
                      children: [
                        Image.file(_image!, height: 300),
                        CustomPaint(
                          size: Size(300 * (_imageSize!.width / _imageSize!.height), 300),
                          painter: BoundingBoxPainter(_highestConfidenceDetection, _imageSize!),
                        ),
                      ],
                    ),
                    SizedBox(height: 20),
                  ],
                  ElevatedButton(
                    onPressed: _pickImage,
                    child: Text('Pick an image'),
                  ),
                  if (_highestConfidenceDetection != null) ...[
                    SizedBox(height: 20),
                    Text('Highest confidence detection:'),
                    Text('Class: ${_highestConfidenceDetection!['class']}, Score: ${_highestConfidenceDetection!['score'].toStringAsFixed(3)}'),
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

class BoundingBoxPainter extends CustomPainter {
  final Map<String, dynamic>? detection;
  final Size originalImageSize;

  BoundingBoxPainter(this.detection, this.originalImageSize);

  @override
  void paint(Canvas canvas, Size size) {
    if (detection == null) return;

    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final scaleX = size.width / 640;
    final scaleY = size.height / 640;

    // final left = detection!['x'] * scaleX;
    // final top = detection!['y'] * scaleY;
    // final right = left + detection!['w'] * scaleX;
    // final bottom = top + detection!['h'] * scaleY;

    // Corrected bounding box calculation
    final centerX = detection!['x'] * scaleX;
    final centerY = detection!['y'] * scaleY;
    final width = detection!['w'] * scaleX;
    final height = detection!['h'] * scaleY;

    final left = centerX - width / 2;
    final top = centerY - height / 2;
    final right = centerX + width / 2;
    final bottom = centerY + height / 2;

    print('left: $left, top: $top, right: $right, bottom: $bottom');

    canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}