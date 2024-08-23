import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';

class ObjectDetectionPage extends StatefulWidget {
  @override
  _LiveObjectDetectionPageState createState() => _LiveObjectDetectionPageState();
}

class _LiveObjectDetectionPageState extends State<ObjectDetectionPage> {
  CameraController? _cameraController;
  OrtSession? _onnxSession;
  Map<String, dynamic>? _highestConfidenceDetection;
  bool _isDetecting = false;
  Size? _previewSize;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeOnnx();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final firstCamera = cameras.first;
    _cameraController = CameraController(
      firstCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController!.initialize();
    if (mounted) {
      setState(() {});
    }
    _startDetection();
  }

  Future<void> _initializeOnnx() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions()..setIntraOpNumThreads(1);
    const assetFileName = 'assets/yolov8n.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _onnxSession = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  void _startDetection() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    _cameraController!.startImageStream((CameraImage image) {
      if (!_isDetecting) {
        _isDetecting = true;
        _runInference(image);
      }
    });
  }

  Future<void> _runInference(CameraImage image) async {
    if (_onnxSession == null) return;
    final run_timer = Stopwatch()..start();

    // final inputData = cameraImageToFloat32List(image);
    final imageData = await image.planes.map((plane) {
                  return plane.bytes;
                }).toList();
                
    // final decodedImage = img.decodeImage(imageData);

    final decodedImage = img.Image.fromBytes(image.width, image.height, imageData[0],
        format: img.Format.rgb);
        // format: img.Format.bgra);

    if (decodedImage == null) return;
    print("Decoding time: ${run_timer.elapsedMilliseconds} ms");

    // setState(() {
    //   _imageSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
    // });

    final resizedImage = img.copyResize(decodedImage, width: 640, height: 640);
    final inputData = _imageToFloat32List(resizedImage);
    print("Preprocessing time: ${run_timer.elapsedMilliseconds} ms");


    final shape = [1, 3, 640, 640];
    final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, shape);

    final inputs = {'images': inputTensor};
    final runOptions = OrtRunOptions();

    try {
      final outputs = await _onnxSession!.run(runOptions, inputs);
      final outputData = outputs[0]!.value as List<List<List<double>>>;
      final highestConfidenceDetection = _processOutput(outputData);
      print("PREDICTION: $highestConfidenceDetection");

      setState(() {
        _highestConfidenceDetection = highestConfidenceDetection;
        _previewSize = Size(image.width.toDouble(), image.height.toDouble());
      });
    } finally {
      inputTensor.release();
      runOptions.release();
      _isDetecting = false;
    } 
    _isDetecting = false;
    print("ATTEMTPED TO RUN INFERENCE");
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


  Float32List _cameraImageToFloat32List(CameraImage image) {
    final float32List = Float32List(3 * 640 * 640);
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < 640; y++) {
      for (int x = 0; x < 640; x++) {
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * 640 + x;

        final yp = image.planes[0].bytes[y * width + x];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        // Convert YUV to RGB
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        // Normalize to [0, 1] and store
        float32List[index] = r / 255.0;
        float32List[index + 640 * 640] = g / 255.0;
        float32List[index + 2 * 640 * 640] = b / 255.0;
      }
    }
    return float32List;
  }
  List<double> cameraImageToFloat32List(CameraImage image) {
    var convertedBytes = Float32List(1 * 3 * image.height * image.width);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var plane in image.planes) {
      var pixels = plane.bytes;

      var stride = plane.bytesPerRow;
      var rows = image.height;
      var cols = image.width;

      for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
          var pixel = pixels[row * stride + col];
          // Normalize the pixel value to [0, 1]
          buffer[pixelIndex++] = pixel / 255.0;
        }
      }
    }

    return convertedBytes;
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
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: Text('Live YOLOv8 Object Detection')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          if (_previewSize != null)
            CustomPaint(
              size: _previewSize!,
              painter: BoundingBoxPainter(_highestConfidenceDetection, _previewSize!),
            ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
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