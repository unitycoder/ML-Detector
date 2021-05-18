/* 
*   NatML Extensions
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.ML.Vision {

    using System;
    using System.Collections.Generic;
    using UnityEngine;
    using Features;
    using Internal;
    using Types;

    /// <summary>
    /// </summary>
    public sealed class TinyYOLOv3Predictor : IMLPredictor<(string label, Rect rect, float score)[]> {

        #region --Client API--
        /// <summary>
        /// Classification labels.
        /// </summary>
        public readonly string[] labels;

        /// <summary>
        /// Create an object detection predictor for the Tiny YOLO v3 model.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="labels">Classification labels.</param>
        public TinyYOLOv3Predictor (MLModel model, string[] labels) {
            this.model = model;
            this.labels = labels;
            // Check
            var classes = (model.outputs[1] as MLArrayType).shape[1];
            if (labels.Length != classes)
                throw new ArgumentOutOfRangeException(nameof(labels), $"YOLO predcitor received {labels.Length} labels but expected {classes}");
        }

        /// <summary>
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public unsafe (string label, Rect rect, float score)[] Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"YOLO predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            if (!(input.type is MLArrayType type))
                throw new ArgumentException(@"YOLO predictor expects an an array or image feature", nameof(inputs));
            // Get size
            var (width, height) = (type.shape[3], type.shape[2]); // Input types are always planar
            var inputSize = new MLArrayFeature<float>(new [] { (float)height, width }, new [] { 1, 2 });
            // Aspect fit input image
            if (input is MLImageFeature imageFeature)
                imageFeature.aspectMode = MLImageFeature.AspectMode.AspectFit;
            // Predict
            var inputType = new MLImageType(416, 416, typeof(float));
            var inputFeature = (input as IMLFeature).Create(inputType);
            var sizeFeature = (inputSize as IMLFeature).Create(inputSize.type);
            var outputFeatures = model.Predict(inputFeature, sizeFeature);  
            inputFeature.ReleaseFeature();
            sizeFeature.ReleaseFeature();          
            // Marshal
            var (boxes, scores, classes) = (outputFeatures[0], outputFeatures[1], outputFeatures[2]);
            var boxesShape = GetShape(boxes);
            var classesShape = GetShape(classes);
            var boxesData = (float*)boxes.FeatureData();
            var scoresData = (float*)scores.FeatureData();
            var classesData = (int*)classes.FeatureData();
            var result = new List<(string, Rect, float)>();
            for (var i = 0; i < classesShape[1]; i++) { // Span<T> support should make this much neater
                var (classIdx, boxIdx) = (classesData[i * 3 + 1], classesData[i * 3 + 2]);
                var top = boxesData[boxIdx * 4];
                var left = boxesData[boxIdx * 4 + 1];
                var bottom = boxesData[boxIdx * 4 + 2];
                var right = boxesData[boxIdx * 4 + 3];
                var label = labels[classIdx];
                var rect = new Rect(left, height - bottom, right - left, bottom - top);
                var confidence = scoresData[classIdx * boxesShape[1] + boxIdx];
                result.Add((label, rect, confidence));
            }
            foreach (var feature in outputFeatures)
                feature.ReleaseFeature();
            // Return
            return result.ToArray();
        }
        #endregion


        #region --Operations--

        private readonly IMLModel model;

        void IDisposable.Dispose () { } // Not used

        private static int[] GetShape (IntPtr feature) {
            feature.FeatureType(out var type);
            var shape = new int[type.FeatureTypeDimensions()];
            type.FeatureTypeShape(shape, shape.Length);
            type.ReleaseFeatureType();
            return shape;
        }
        #endregion
    }
}