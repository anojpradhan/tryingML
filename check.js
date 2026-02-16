let video;
let handpose;
let predictions = [];

let neuralNetwork;

let label = "Waiting...";
let confidence = 0;

let modelLoaded = false;
let isClassifying = false;

function setup() {
  createCanvas(640, 480);

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  handpose = ml5.handpose(video, () => {
    console.log("Handpose ready!");
  });

  handpose.on("predict", (results) => {
    predictions = results;
  });

  neuralNetwork = ml5.neuralNetwork({ task: "classification" });

  neuralNetwork.load(
    {
      model: "model.json",
      metadata: "model_meta.json",
      weights: "model.weights.bin",
    },
    () => {
      console.log("Custom model loaded!");
      modelLoaded = true;
    }
  );
}

function draw() {
  image(video, 0, 0, width, height);

  if (predictions.length > 0) {
    let hand = predictions[0];

    drawKeypoints(hand);

    if (modelLoaded && !isClassifying) {
      classifyHand(hand);
    }
  }

  drawPrediction();
}

function classifyHand(hand) {
  isClassifying = true;

  let inputs = extractKeypoints(hand);

  if (inputs.length !== 63) {
    isClassifying = false;
    return;
  }

  neuralNetwork.classify(inputs, (error, results) => {
    if (error) {
      console.error(error);
      isClassifying = false;
      return;
    }

    if (!results || results.length === 0) {
      isClassifying = false;
      return;
    }

    if (!results[0].confidence || isNaN(results[0].confidence)) {
      console.log("⚠️ Confidence invalid");
      isClassifying = false;
      return;
    }

    label = results[0].label;
    confidence = results[0].confidence;

    console.log(label, confidence);

    isClassifying = false;
  });
}

function extractKeypoints(hand) {
  let inputs = [];

  let wristX = hand.landmarks[0][0];
  let wristY = hand.landmarks[0][1];

  for (let i = 0; i < hand.landmarks.length; i++) {
    let x = hand.landmarks[i][0] - wristX;
    let y = hand.landmarks[i][1] - wristY;
    let z = hand.landmarks[i][2];

    inputs.push(x);
    inputs.push(y);
    inputs.push(z);
  }

  return inputs;
}

function drawKeypoints(hand) {
  for (let i = 0; i < hand.landmarks.length; i++) {
    let x = hand.landmarks[i][0];
    let y = hand.landmarks[i][1];

    fill(0, 255, 0);
    noStroke();
    circle(x, y, 8);
  }
}

function drawPrediction() {
  fill(255);
  textSize(24);
  textAlign(LEFT);

  text("Prediction: " + label, 10, height - 50);

  if (!isNaN(confidence)) {
    text(
      "Confidence: " + (confidence * 100).toFixed(2) + "%",
      10,
      height - 20
    );
  }
}
