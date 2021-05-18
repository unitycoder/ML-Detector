# ML Detection

![detector](.media/cows.png)

This is a library for classifying images in Unity Engine using NatML. The following ML models are included:
- [TinyYOLO v3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3)

## Setup Instructions
This package requires NatML, so make sure NatML is imported into your project. Then in your project's `Packages/manifest.json` file, add the following:
```json
{
  "dependencies": {
    "com.natsuite.ml.detection": "git+https://github.com/natsuite/ML-Detection"
  }
}
```

## Detecting Objects in an Image
First, assign the object detection model (in the `ML` folder) to an `MLModelData` field in your script:
```csharp
using NatSuite.ML;
using NatSuite.ML.Vision;

public class Classifier : MonoBehaviour {

    public MLModelData modelData; // Assign this in the Inspector
}
```

Then create a detection predictor corresponding to the model:
```csharp
void Start () {
    var model = modelData.Deserialize();
    var predictor = new TinyYOLOv3Predictor(model, modelData.labels);
}
```

Detect objects in an image:
```csharp
Texture2D image = ...;
(string label, Rect rect, float score)[] detections = predictor.Predict(image);
```

## Visualizing Detections
*INCOMPLETE*

## Requirements
- Unity 2019.2+
- NatML 1.0+

## Supported Platforms
- Android 7.0 Nougat or newer (API level 24+)
- iOS 13+
- macOS 10.15+
- Windows 10 64-bit

## Quick Tips
- See the [NatML documentation](https://docs.natsuite.io/natml).
- Join the [NatSuite community on Discord](https://discord.gg/y5vwgXkz2f).
- See [NatML on Unity Forums](https://forum.unity.com/threads/open-beta-natml-machine-learning-runtime.1109339/).
- Contact us at [hi@natsuite.io](mailto:hi@natsuite.io).