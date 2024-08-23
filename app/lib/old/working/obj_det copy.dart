
import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:camera/camera.dart';
import 'package:audioplayers/audioplayers.dart';



class ObjectDetectionPage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const ObjectDetectionPage({Key? key, required this.cameras}) : super(key: key);

  @override
  _ObjectDetectionPageState createState() => _ObjectDetectionPageState();
}





const RESOLUTION_SIZE = Size(480, 720);

































class _ObjectDetectionPageState extends State<ObjectDetectionPage> {
  static AudioPlayer player = new AudioPlayer();
  late FlutterVision vision;
  late CameraController _cameraController;
  List<Map<String, dynamic>>? _recognitions;
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeVision();
  }

  Future<void> _initializeCamera() async {
    // _cameraController = CameraController(widget.cameras[0], ResolutionPreset.low);
    _cameraController = CameraController(widget.cameras[0], ResolutionPreset.medium);
    await _cameraController.initialize();
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _initializeVision() async {
    vision = FlutterVision();
    await vision.loadYoloModel(
      labels: 'assets/labels.txt',
      modelPath: 'assets/1.tflite',
      modelVersion: "yolov5",
      quantization: true,
      numThreads: 2,
      useGpu: true,
    );
  }

  Future<void> _captureAndDetect() async {
    if (_isDetecting) return;
    _isDetecting = true;

    // start timer
    Stopwatch stopwatch = new Stopwatch()..start();
    try {
      final image = await _cameraController.takePicture();
      Stopwatch stopwatch_d = new Stopwatch()..start();
      print('Cam time: ${stopwatch.elapsed.inMilliseconds}');
      final result = await vision.yoloOnImage(
        bytesList: await image.readAsBytes(),
        imageHeight: _cameraController.value.previewSize!.height.toInt(),
        imageWidth: _cameraController.value.previewSize!.width.toInt(),
        iouThreshold: 0.4,
        confThreshold: 0.4,
        classThreshold: 0.5,
      );
      print('DET in ${stopwatch_d.elapsed.inMilliseconds}');
      for (var result_item in result) {
        var box = result_item['box'];
        print(result_item['box']);
        result_item['tag'];

        if (result_item['tag']=="person"){
          if ((480.0-box[3])<240){
            if (((480.0-box[3])+(box[3]-box[1]))>240){
              if (box[0]<360){
                if (box[0]+(box[2]-box[0])>360){
                  print("HIT THE TARGET YA");
                  print('doSomething() executed in ${stopwatch.elapsed.inMilliseconds}');
                  var alarmAudioPath = "hitfast.mp3";
                  await player.play(AssetSource(alarmAudioPath));
                }
              }
            }
          }
        }
      }

      setState(() {
        _recognitions = result;
      });
    } catch (e) {
      print("Error processing image: $e");
    } finally {
      _isDetecting = false;
    }
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
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: ElevatedButton(
                onPressed: _captureAndDetect,
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

    // make a small circle in the dead center of the image
    final center = Offset(size.width / 2, size.height / 2);
    canvas.drawCircle(center, 5.0, paint);
  }

  Rect _convertCoordinates(List<dynamic> box, Size size) {
    final double x = 480.0-box[3];
    final double y = box[0];
    final double w = (box[3]-box[1]);
    final double h = box[2]-box[0];

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