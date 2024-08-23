import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:ui' as ui;

class ObjectDetectionPage extends StatefulWidget {
  @override
  _ObjectDetectionPageState createState() => _ObjectDetectionPageState();
}

class _ObjectDetectionPageState extends State<ObjectDetectionPage> {
  late FlutterVision vision;
  File? _image;
  List<Map<String, dynamic>>? _recognitions;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _initializeVision();
  }

  Future<void> _initializeVision() async {
    vision = FlutterVision();
    await vision.loadYoloModel(
      labels: 'assets/labels.txt',
      // modelPath: 'assets/yolov8n_float16.tflite',
      modelPath: 'assets/yolov8n_float32.tflite',
      modelVersion: "yolov8",
      quantization: false,
      numThreads: 1,
      useGpu: true,
    );
    setState(() {
      _loading = false;
    });
  }

  Future<void> _pickImage() async {
    final ImagePicker _picker = ImagePicker();
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    setState(() {
      _loading = true;
      _image = File(image.path);
    });
    await _detectObjects();
  }

  Future<void> _detectObjects() async {
    if (_image == null) return;
    
    final results = await vision.yoloOnImage(
      bytesList: await _image!.readAsBytes(),
      // imageHeight: 640,
      // imageWidth: 640,
      imageHeight: 2336,
      imageWidth: 1080,
      iouThreshold: 0.5,
      confThreshold: 0.3,
      classThreshold: 0.5,
    );
    print("DETECTING");
    print(results);
    
    setState(() {
      _recognitions = results;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLOv8 Object Detection'),
      ),
      body: _loading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Column(
                children: [
                  if (_image != null) ...[
                    Stack(
                      children: [
                        Image.file(_image!),
                        if (_recognitions != null)
                          BoundingBoxPainter(
                            image: _image!,
                            recognitions: _recognitions!,
                          ),
                      ],
                    ),
                    SizedBox(height: 20),
                  ],
                  ElevatedButton(
                    onPressed: _pickImage,
                    child: Text('Pick an image'),
                  ),
                  SizedBox(height: 20),
                  if (_recognitions != null)
                    Column(
                      children: _recognitions!.map((recognition) {
                        return ListTile(
                          title: Text(recognition['tag']),
                          subtitle: Text('Confidence: ${(recognition['box'][4] * 100).toStringAsFixed(2)}%'),
                        );
                      }).toList(),
                    ),
                ],
              ),
            ),
    );
  }

  @override
  void dispose() {
    vision.closeYoloModel();
    super.dispose();
  }
}

class BoundingBoxPainter extends StatelessWidget {
  final File image;
  final List<Map<String, dynamic>> recognitions;

  BoundingBoxPainter({required this.image, required this.recognitions});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ui.Image>(
      future: _loadImage(image),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done && snapshot.data != null) {
          return CustomPaint(
            size: Size(snapshot.data!.width.toDouble(), snapshot.data!.height.toDouble()),
            painter: BoxPainter(recognitions: recognitions, image: snapshot.data!),
          );
        } else {
          return Container();
        }
      },
    );
  }

  Future<ui.Image> _loadImage(File file) async {
    final data = await file.readAsBytes();
    return await decodeImageFromList(data);
  }
}

class BoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> recognitions;
  final ui.Image image;

  BoxPainter({required this.recognitions, required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.red;

    for (var recognition in recognitions) {
      final box = recognition['box'];
      print(box);
      // final left = box[0]* size.width;
      // final top = box[1] * size.height;
      // final right = box[2] * size.width;
      // final bottom = box[3] * size.height;
      final left = box[0];
      final top = box[1];
      final right = box[2];
      final bottom = box[3];

      canvas.drawRect(
        Rect.fromLTRB(left, top, right, bottom),
        paint,
      );
      canvas.drawRect(
        Rect.fromLTRB(10, 10, 100, 100),
        paint,
      );

      final textPainter = TextPainter(
        text: TextSpan(
          text: "${recognition['tag']} ${(box[4] * 100).toStringAsFixed(0)}%",
          style: TextStyle(color: Colors.red, fontSize: 14),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(left, top - 20));
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}