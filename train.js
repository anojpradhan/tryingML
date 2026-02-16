let video;
let handpose;
let predictions = [];

let neuralNetwork;

let collecting = false;
let targetLabel = "";

let thumbsCount = 0;
let normalCount = 0;

// Number of points * 2 (x,y)
const INPUTS = 42;

function setup() {
  createCanvas(640, 480);

  // Setup webcam
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Load Handpose model
  handpose = ml5.handpose(video, () => {
    console.log("Handpose model loaded!");
  });

  handpose.on("predict", (results) => {
    predictions = results;
  });

  // Setup Neural Network
  const options = {
    inputs: INPUTS, // 21 points * 2 (x,y)
    outputs: 2,
    task: "classification",
    debug: true,
  };

  neuralNetwork = ml5.neuralNetwork(options);
}

function draw() {
  image(video, 0, 0, width, height);

  if (predictions.length > 0) {
    let hand = predictions[0];

    drawKeypoints(hand);

    if (collecting) {
      let inputs = extractKeypoints(hand);

      // Skip invalid inputs
      if (inputs.length !== INPUTS || inputs.some((v) => isNaN(v))) {
        return;
      }

      neuralNetwork.addData(inputs, { label: targetLabel });

      if (targetLabel === "thumbs_up") thumbsCount++;
      if (targetLabel === "normal") normalCount++;

      console.log("Thumbs:", thumbsCount, "Normal:", normalCount);
    }
  }

  drawStats();
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

function extractKeypoints(hand) {
  let inputs = [];

  let wristX = hand.landmarks[0][0];
  let wristY = hand.landmarks[0][1];

  for (let i = 0; i < hand.landmarks.length; i++) {
    let x = hand.landmarks[i][0] - wristX;
    let y = hand.landmarks[i][1] - wristY;
    inputs.push(x);
    inputs.push(y);
    // z skipped for stability
  }

  return inputs;
}

function keyPressed() {
  if (key === "T") {
    targetLabel = "thumbs_up";
    collecting = true;
    console.log("Collecting THUMBS UP");
  }

  if (key === "N") {
    targetLabel = "normal";
    collecting = true;
    console.log("Collecting NORMAL");
  }

  if (keyCode === 32) {
    // SPACE stops collection
    collecting = false;
    console.log("Stopped collecting");
  }

  if (key === "S") {
    collecting = false;

    if (thumbsCount < 50 || normalCount < 50) {
      console.log(" Collect at least 50 varied samples per class!");
      return;
    }

    console.log("Training started...");

    neuralNetwork.normalizeData();

    neuralNetwork.train({ epochs: 60, batchSize: 16 }, finishedTraining);
  }
}

function finishedTraining() {
  console.log(" Training complete!");
  neuralNetwork.save(); // Downloads model.json, model_meta.json, model.weights.bin
}

function drawStats() {
  fill(255);
  textSize(16);
  text("Thumbs: " + thumbsCount, 10, height - 40);
  text("Normal: " + normalCount, 10, height - 20);
}
