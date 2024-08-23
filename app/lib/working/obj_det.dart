import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:camera/camera.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;


class ObjectDetectionPage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const ObjectDetectionPage({Key? key, required this.cameras}) : super(key: key);

  @override
  _ObjectDetectionPageState createState() => _ObjectDetectionPageState();
}

class _ObjectDetectionPageState extends State<ObjectDetectionPage> {
  static AudioPlayer player = new AudioPlayer();
  late FlutterVision vision;
  late CameraController _cameraController;
  List<Map<String, dynamic>>? _recognitions;
  bool _isDetecting = false;
  bool _buttonPressed = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeVision();
  }

  Future<void> _initializeCamera() async {
    _cameraController = CameraController(widget.cameras[0], ResolutionPreset.medium);
    await _cameraController.initialize();
    if (mounted) {
      setState(() {});
    }

    await _cameraController.startImageStream((image) async {
      if (!_isDetecting){
        _processCameraFrame(image);
      }
      // if (_buttonPressed){
      //   if (!_isDetecting){
      //     _processCameraFrame(image);
      //   }
      // }
    });
  }

  Future<void> _initializeVision() async {
    vision = FlutterVision();
    await vision.loadYoloModel(
      labels: 'assets/labels.txt',
      // modelPath: 'assets/1.tflite',
      // modelVersion: "yolov5",
      modelPath: 'assets/yolov8n_float16.tflite',
      modelVersion: "yolov8",
      quantization: false,
      numThreads: 1,
      useGpu: true,
    );
  }

  Future<void> _processCameraFrame(CameraImage image) async {
    _isDetecting = true;
    _buttonPressed = false;
    try {
      // final result = await vision.yoloOnFrame(\
      Stopwatch stopwatch = new Stopwatch()..start();
      var img_file = await convertCameraImageToXFile(image);
      var image_bytes=await img_file?.readAsBytes();
      var result_status=[];
      print('Cam time: ${stopwatch.elapsed.inMilliseconds}');
      
      if (image_bytes != null) {
        final result = await vision.yoloOnImage(
          bytesList: image_bytes,
          //bytesList: image.planes.map((plane) => plane.bytes).toList(),
          //bytesList: convertCameraImageToBytesList(image),//.planes.map((plane) => plane.bytes).toList(),
          imageHeight: (image.height/2).toInt(),
          imageWidth: (image.width/2).toInt(),
          // iouThreshold: 0.0,
          // confThreshold: 0.0,
          // classThreshold: 0.0,
          // iouThreshold: 0.2,
          // confThreshold: 0.2,
          // classThreshold: 0.3,
          iouThreshold: 0.3,
          confThreshold: 0.3,
          classThreshold: 0.4,
        );
        result_status = result;
        for (var result_item in result) {
          var box = result_item['box'];
          if (result_item['tag'] == "person") {
            if ((480.0 - box[3]) < 240) {
              if (((480.0 - box[3]) + (box[3] - box[1])) > 240) {
                if (box[0] < 360) {
                  if (box[0] + (box[2] - box[0]) > 360) {
                    print("HIT THE TARGET YA");
                    var alarmAudioPath = "hitfast.mp3";
                    await player.play(AssetSource(alarmAudioPath));
                  }
                }
              }
            }
          }
        }
      } else {
        result_status = [];
      }
      print('Finish TIme: ${stopwatch.elapsed.inMilliseconds}');

      if (result_status.isEmpty) {
        print('No objects detected');
      } else {
        print('Detections: $result_status');
      }

      setState(() {
        _recognitions = result_status.cast<Map<String, dynamic>>();
      });
    } catch (e) {
      print("Error processing frame: $e");
    } finally {
      _isDetecting = false;
    }
  }

  List<Uint8List> convertCameraImageToBytesList(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final img.Image imgBuffer = img.Image(width, height);

    // Converting YUV420 to RGB
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = ((y >> 1) * (width >> 1)) + (x >> 1);
        final int index = y * width + x;

        final int yValue = image.planes[0].bytes[index];
        final int uValue = image.planes[1].bytes[uvIndex];
        final int vValue = image.planes[2].bytes[uvIndex];

        // Convert YUV to RGB
        final r = (yValue + (1.370705 * (vValue - 128))).clamp(0, 255).toInt();
        final g = (yValue - (0.337633 * (uValue - 128)) - (0.698001 * (vValue - 128))).clamp(0, 255).toInt();
        final b = (yValue + (1.732446 * (uValue - 128))).clamp(0, 255).toInt();

        // Set the pixel in the image
        imgBuffer.setPixel(x, y, img.getColor(r, g, b));
      }
    }

    // Convert the image to bytes (PNG format)
    Uint8List pngBytes = Uint8List.fromList(img.encodePng(imgBuffer));
    
    // Create a List<Uint8List> with the PNG bytes as the only element
    return [pngBytes];
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(title: Text('Object Detection')),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          if (_recognitions != null)
            CustomPaint(
              painter: BoundingBoxPainter(_recognitions!, _cameraController.value),
              child: Container(),
            ),
          // a button to capture and detect
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: ElevatedButton(
                onPressed: () async {
                  _buttonPressed = true;
                  await Future.delayed(Duration(seconds: 1));
                },
                child: Text('Capture and Detect'),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    vision.closeYoloModel();
    super.dispose();
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> recognitions;
  final CameraValue cameraValue;

  BoundingBoxPainter(this.recognitions, this.cameraValue);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;

    for (var result in recognitions) {
      if (result["tag"]=="person"){
        final rect = _convertCoordinates(result['box'], size);
        canvas.drawRect(rect, paint);

        final textPainter = TextPainter(
          text: TextSpan(
            text: "${result['tag']} ${(result['box'][4] * 100).toStringAsFixed(0)}%",
            style: TextStyle(color: Colors.red, fontSize: 14),
          ),
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
        textPainter.paint(canvas, rect.topLeft);
      }

      final center = Offset(size.width / 2, size.height / 2);
      canvas.drawCircle(center, 5.0, paint);
    }
  }

  Rect _convertCoordinates(List<dynamic> box, Size size) {
    

    // final double x = 480.0 - box[3];
    final double x = 360.0 - (box[3]/2);
    final double y = 180.0 + (box[0]/2);
    final double w = (box[3] - box[1])/2;
    final double h = 100.0;//(box[2] - box[0])/2;

    final double scaleX = size.width / cameraValue.previewSize!.height;
    final double scaleY = size.height / cameraValue.previewSize!.width;

    return Rect.fromLTWH(
      x * scaleX,
      y * scaleY,
      w * scaleX,
      h * scaleY,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}


































Future<XFile?> convertCameraImageToXFile(CameraImage image, {double zoom = 2.0}) async {
  try {
    // Convert YUV420 format to RGB
    final rgbImage = _convertYUV420ToRGB(image);
    
    // Apply zoom to the center of the image
    final zoomedImage = _zoomIntoCenter(rgbImage, zoom);
    
    // Encode the image to PNG format
    final pngBytes = img.encodePng(zoomedImage);
    
    // Create a temporary file and write the bytes
    final tempDir = await getTemporaryDirectory();
    final tempFile = File('${tempDir.path}/temp_image_${DateTime.now().millisecondsSinceEpoch}.png');
    await tempFile.writeAsBytes(pngBytes);
    
    // Create XFile from the temporary file
    return XFile(tempFile.path);
  } catch (e) {
    print('Error converting CameraImage to XFile: $e');
    return null;
  }
}

img.Image _convertYUV420ToRGB(CameraImage image) {
  final width = image.width;
  final height = image.height;
  final yRowStride = image.planes[0].bytesPerRow;
  final uvRowStride = image.planes[1].bytesPerRow;
  final uvPixelStride = image.planes[1].bytesPerPixel!;

  final rgbImage = img.Image(width, height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
      final int index = yRowStride * y + x;

      final yp = image.planes[0].bytes[index];
      final up = image.planes[1].bytes[uvIndex];
      final vp = image.planes[2].bytes[uvIndex];

      int r = (yp + vp * 1436 ~/ 1024 - 179).clamp(0, 255);
      int g = (yp - up * 46549 ~/ 131072 + 44 - vp * 93604 ~/ 131072 + 91).clamp(0, 255);
      int b = (yp + up * 1814 ~/ 1024 - 227).clamp(0, 255);

      rgbImage.setPixel(x, y, img.getColor(r, g, b));
    }
  }

  return rgbImage;
}

img.Image _zoomIntoCenter(img.Image image, double zoom) {
  if (zoom <= 1.0) return image;

  int newWidth = (image.width / zoom).round();
  int newHeight = (image.height / zoom).round();

  int startX = (image.width - newWidth) ~/ 2;
  int startY = (image.height - newHeight) ~/ 2;

  img.Image zoomedImage = img.copyCrop(
    image, startX, startY, newWidth, newHeight,
  );

  return img.copyResize(zoomedImage, width: image.width, height: image.height);
}