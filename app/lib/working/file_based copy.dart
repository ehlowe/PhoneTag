import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';


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
      // modelPath: 'assets/1.tflite',
      // modelVersion: "yolov5",
      modelPath: 'assets/yolov8n_float16.tflite',
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
    print("DETECTING");
    if (_image == null) return;
    
    Stopwatch stopwatch = new Stopwatch()..start();
    final results = await vision.yoloOnImage(
      bytesList: await _image!.readAsBytes(),
      imageHeight: 640,
      imageWidth: 640,
      iouThreshold: 0.5,
      confThreshold: 0.3,
      classThreshold: 0.5,
    );

    print('Detection took ${stopwatch.elapsed.inMilliseconds}ms');
    // print(results);
    print("GOT RESULT");
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
        title: Text('YOLOv5 Object Detection'),
      ),
      body: _loading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Column(
                children: [
                  if (_image != null) ...[
                    Image.file(_image!),
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













