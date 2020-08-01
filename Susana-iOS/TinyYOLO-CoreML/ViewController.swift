import UIKit
import Vision
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!

  // true: use Vision to drive Core ML, false: use plain Core ML
  let useVision = false

  // Disable this to see the energy impact of just running the neural net,
  // otherwise it also counts the GPU activity of drawing the bounding boxes.
  let drawBoundingBoxes = true

  // How many predictions we can do concurrently.
  static let maxInflightBuffers = 3

  let yolo = YOLO()

  var videoCapture: VideoCapture!
  var requests = [VNCoreMLRequest]()
  var startTimes: [CFTimeInterval] = []

  var boundingBoxes = [BoundingBox]()
  var colors: [UIColor] = []
  var dangerCount:Int = 0
  var safeCount:Int = 0

  let ciContext = CIContext()
  var resizedPixelBuffers: [CVPixelBuffer?] = []

  var framesDone = 0
  var frameCapturingStartTime = CACurrentMediaTime()

  var inflightBuffer = 0
  let semaphore = DispatchSemaphore(value: ViewController.maxInflightBuffers)

  override func viewDidLoad() {
    super.viewDidLoad()

    timeLabel.text = ""

    setUpBoundingBoxes()
    setUpCoreImage()
    setUpVision()
    setUpCamera()

    frameCapturingStartTime = CACurrentMediaTime()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - Initialization

  func setUpBoundingBoxes() {
    for _ in 0..<YOLO.maxBoundingBoxes {
      boundingBoxes.append(BoundingBox())
    }

    // Make colors for the bounding boxes. There is one color for each class,
    // 20 classes in total.
    /*
    for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
      for g: CGFloat in [0.3, 0.7] {
        for b: CGFloat in [0.4, 0.8] {
          let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          colors.append(color)
        }
      }
    }
    */
  }

  func setUpCoreImage() {
    // Since we might be running several requests in parallel, we also need
    // to do the resizing in different pixel buffers or we might overwrite a
    // pixel buffer that's already in use.
    for _ in 0..<ViewController.maxInflightBuffers {
      var resizedPixelBuffer: CVPixelBuffer?
      let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                       kCVPixelFormatType_32BGRA, nil,
                                       &resizedPixelBuffer)

      if status != kCVReturnSuccess {
        print("Error: could not create resized pixel buffer", status)
      }
      resizedPixelBuffers.append(resizedPixelBuffer)
    }
  }

  func setUpVision() {
    guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
      print("Error: could not create Vision model")
      return
    }

    for _ in 0..<ViewController.maxInflightBuffers {
      let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)

      // NOTE: If you choose another crop/scale option, then you must also
      // change how the BoundingBox objects get scaled when they are drawn.
      // Currently they assume the full input image is used.
      request.imageCropAndScaleOption = .scaleFill
      requests.append(request)
    }
  }

  func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self
    videoCapture.desiredFrameRate = 240
    videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { success in
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.resizePreviewLayer()
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxes {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

  // MARK: - Doing inference

  func predict(image: UIImage) {
    if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
      predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
    }
  }

  func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
    // Measure how long it takes to predict a single video frame.
    let startTime = CACurrentMediaTime()

    // This is an alternative way to resize the image (using vImage):
    //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
    //                                              width: YOLO.inputWidth,
    //                                              height: YOLO.inputHeight) {

    // Resize the input with Core Image to 416x416.
    if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
      let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
      let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
      let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
      let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
      let scaledImage = ciImage.transformed(by: scaleTransform)
      ciContext.render(scaledImage, to: resizedPixelBuffer)

      // Give the resized input to our model.
      if let boundingBoxes = yolo.predict(image: resizedPixelBuffer) {
        let elapsed = CACurrentMediaTime() - startTime
        showOnMainThread(boundingBoxes, elapsed)
      } else {
        print("BOGUS")
      }
    }

    self.semaphore.signal()
  }

  func predictUsingVision(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
    // Measure how long it takes to predict a single video frame. Note that
    // predict() can be called on the next frame while the previous one is
    // still being processed. Hence the need to queue up the start times.
    startTimes.append(CACurrentMediaTime())

    // Vision will automatically resize the input image.
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    let request = requests[inflightIndex]

    // Because perform() will block until after the request completes, we
    // run it on a concurrent background queue, so that the next frame can
    // be scheduled in parallel with this one.
    DispatchQueue.global().async {
      try? handler.perform([request])
    }
  }

  func visionRequestDidComplete(request: VNRequest, error: Error?) {
    if let observations = request.results as? [VNCoreMLFeatureValueObservation],
       let features = observations.first?.featureValue.multiArrayValue {

      let boundingBoxes = yolo.computeBoundingBoxes(features: features)
      let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
      showOnMainThread(boundingBoxes, elapsed)
    } else {
      print("BOGUS!")
    }

    self.semaphore.signal()
  }

  func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
    if drawBoundingBoxes {
      DispatchQueue.main.async {
        // For debugging, to make sure the resized CVPixelBuffer is correct.
        //var debugImage: CGImage?
        //VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
        //self.debugImageView.image = UIImage(cgImage: debugImage!)

        self.show(predictions: boundingBoxes)

        let fps = self.measureFPS()
        self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
      }
    }
  }

  func measureFPS() -> Double {
    // Measure how many frames were actually delivered per second.
    framesDone += 1
    let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
    let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
    if frameCapturingElapsed > 1 {
      framesDone = 0
      frameCapturingStartTime = CACurrentMediaTime()
    }
    return currentFPSDelivered
  }

  func show(predictions: [YOLO.Prediction]) {
    var _color = UIColor.green
    //var prediction: YOLO.Prediction
    var rects = [CGRect]()
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        //prediction = predictions[i]
        print("I val: ",i,". Pred: ",predictions.count,". Name: ", labels[predictions[i].classIndex],". Size: ",rects.count)
        // The predicted bounding box is in the coordinate space of the input
        // image, which is a square image of 416x416 pixels. We want to show it
        // on the video preview, which is as wide as the screen and has a 16:9
        // aspect ratio. The video preview also may be letterboxed at the top
        // and bottom.
        let width = view.bounds.width
        let height = width * 16 / 9
        let scaleX = width / CGFloat(YOLO.inputWidth)
        let scaleY = height / CGFloat(YOLO.inputHeight)
        let top = (view.bounds.height - height) / 2

        // Translate and scale the rectangle to our own coordinate system.
        //rects[i] = predictions[i].rect
        
        rects.insert(predictions[i].rect, at: i)
        rects[i].origin.x *= scaleX
        rects[i].origin.y *= scaleY
        rects[i].origin.y += top
        rects[i].size.width *= scaleX
        rects[i].size.height *= scaleY
        
        var rect1 = predictions[i].rect
        rect1.origin.x *= scaleX
        rect1.origin.y *= scaleY
        rect1.origin.y += top
        rect1.size.width *= scaleX
        rect1.size.height *= scaleY
        
        // Show the bounding box.        
        let predLabel = labels[predictions[i].classIndex]
        //let label = String(format: "%@ %.1f", predLabel, predictions[i].score * 100)
        if predLabel == "person"{
            if ( (predictions.count-1 > i) && (labels[predictions[i+1].classIndex] == "person") ){
                rects.insert(predictions[i+1].rect, at: i+1)
                rects[i+1].origin.x *= scaleX
                rects[i+1].origin.y *= scaleY
                rects[i+1].origin.y += top
                rects[i+1].size.width *= scaleX
                rects[i+1].size.height *= scaleY
                if isClose(p1: rects[i], p2: rects[i+1], angle_factor: 0.8) != 0{
                    _color = UIColor.red
                    let x1 = rects[i].origin.x + (rects[i].size.width / 2)
                    let y1 = rects[i].origin.y + (rects[i].size.height / 2)
                    let x2 = rects[i+1].origin.x + (rects[i+1].size.width / 2)
                    let y2 = rects[i+1].origin.y + (rects[i+1].size.height / 2)
                    boundingBoxes[i].showLine(fromPoint: CGPoint(x: x1, y: y1), toPoint: CGPoint(x: x2, y: y2))
                    boundingBoxes[i].show(frame: rects[i], label: "DANGER", color: _color)
                    boundingBoxes[i+1].show(frame: rects[i+1], label: "DANGER", color: _color)
                }
                else {
                    _color = UIColor.green
                    boundingBoxes[i].hideLine()
                    boundingBoxes[i].hide()
                    boundingBoxes[i+1].hide()
                }
                //boundingBoxes[i].show(frame: rects[i], label: "SAFE", color: _color)
            }
            else{
                boundingBoxes[i].show(frame: rects[i], label: "SAFE", color: _color)
            }
        }
      } else {
        boundingBoxes[i].hide()
      }
    }
  }
    
    func dist(c1: CGPoint,c2: CGPoint) -> CGFloat{
        let x = (c1.x - c2.x) * (c1.x - c2.x)
        let y = (c1.y - c2.y) * (c1.y - c2.y)
        return  sqrt(x + y)
    }
    
    func T2S(T: CGFloat) -> CGFloat{
        let S = abs(T / sqrt(1+(T*T)) )
        return S
    }
    func T2C(T: CGFloat) -> CGFloat{
        let C = abs(1 / sqrt(1+(T*T)) )
        return C
    }
    
    func isClose(p1: CGRect, p2: CGRect, angle_factor: CGFloat) -> Int{
        let x1 = p1.origin.x + (p1.size.width / 2)
        let y1 = p1.origin.y + (p1.size.height / 2)
        let x2 = p2.origin.x + (p2.size.width / 2)
        let y2 = p2.origin.y + (p2.size.height / 2)
        let _c1 = CGPoint(x: x1, y: y1)
        let _c2 = CGPoint(x: x2, y: y2)
        let centreDistance = dist(c1: _c1, c2: _c2)
        var a_w: CGFloat
        var a_h: CGFloat
        var T: CGFloat = 0
        let S: CGFloat
        let C: CGFloat
        
        if p1.size.height < p2.size.height {
            a_w = p1.size.width
            a_h = p1.size.height
        }
        else {
            a_w = p2.size.width
            a_h = p2.size.height
        }
        
        let epsilon: CGFloat = 1e-8
        T = (_c2.y - _c1.y)/(_c2.x - _c1.x + epsilon)
        
        S = T2S(T: T)
        C = T2C(T: T)
        
        let d_hor = C * centreDistance
        let d_ver = S * centreDistance
        let vc_calib_hor = a_w * 1.3
        let vc_calib_ver = a_h * 0.4 * angle_factor
        let c_calib_hor = a_w * 1.7
        let c_calib_ver = a_h * 0.2 * angle_factor
        
        if ((0 < d_hor) && (d_hor < vc_calib_hor) && (0 < d_ver) && (d_ver < vc_calib_ver)){
            return 1
        }
        else if ((0 < d_hor) && (d_hor < c_calib_hor) && (0 < d_ver) && (d_ver < c_calib_ver)){
            return 2
        }
        else{
            return 0
        }
    }
}

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
    // For debugging.
    //predict(image: UIImage(named: "dog416")!); return

    if let pixelBuffer = pixelBuffer {
      // The semaphore will block the capture queue and drop frames when
      // Core ML can't keep up with the camera.
      semaphore.wait()

      // For better throughput, we want to schedule multiple prediction requests
      // in parallel. These need to be separate instances, and inflightBuffer is
      // the index of the current request.
      let inflightIndex = inflightBuffer
      inflightBuffer += 1
      if inflightBuffer >= ViewController.maxInflightBuffers {
        inflightBuffer = 0
      }

      if useVision {
        // This method should always be called from the same thread!
        // Ain't nobody likes race conditions and crashes.
        self.predictUsingVision(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
      } else {
        // For better throughput, perform the prediction on a concurrent
        // background queue instead of on the serial VideoCapture queue.
        DispatchQueue.global().async {
          self.predict(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
        }
      }
    }
  }
}
