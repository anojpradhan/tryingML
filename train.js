let handPose;
let video;
let hands = [];
let neuralNetwork;
let targetLabel = "";
let collecting = false;

function preload() {
  handPose = ml5.handPose();
}

function setup() {
  createCanvas(640, 480);

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  handPose.detectStart(video, gotHands);

  // Neural Network setup
  let options = {
    inputs: 63,
    outputs: 2,
    task: "classification",
    debug: true,
  };

  neuralNetwork = ml5.neuralNetwork(options);
}

function gotHands(results) {
  hands = results;
}

function draw() {
  image(video, 0, 0, width, height);

  if (hands.length > 0) {
    let hand = hands[0];

    // Draw keypoints
    for (let i = 0; i < hand.keypoints.length; i++) {
      let kp = hand.keypoints[i];
      fill(0, 255, 0);
      noStroke();
      circle(kp.x, kp.y, 8);
    }

    // Collect data continuously if collecting is true
    if (collecting) {
      let inputs = extractKeypoints(hand);
      neuralNetwork.addData(inputs, { label: targetLabel });
      console.log("Collecting:", targetLabel);
    }
  }
}

function keyPressed() {
  // Start collecting thumbs up
  if (key === "t" || key === "T") {
    targetLabel = "thumbs_up";
    collecting = true;
    console.log("Collecting THUMBS UP...");
  }

  // Start collecting normal hand
  if (key === "n" || key === "N") {
    targetLabel = "normal";
    collecting = true;
    console.log("Collecting NORMAL...");
  }

  // Stop collecting
  if (key === " ") {
    collecting = false;
    console.log("Stopped collecting.");
  }

  // Train model
  if (key === "s" || key === "S") {
    collecting = false;
    console.log("Training started...");

    neuralNetwork.normalizeData();

    neuralNetwork.train({ epochs: 50 }, finishedTraining);
  }
}

function finishedTraining() {
  console.log("Training complete!");
  neuralNetwork.save(); // downloads model files
}

// Extract 63 inputs (21 keypoints × x,y,z)
function extractKeypoints(hand) {
  let inputs = [];

  // Use wrist as reference (IMPORTANT for stability)
  let wrist = hand.keypoints[0];

  for (let i = 0; i < hand.keypoints.length; i++) {
    let kp = hand.keypoints[i];

    inputs.push(kp.x - wrist.x);
    inputs.push(kp.y - wrist.y);
    inputs.push(kp.z);
  }

  return inputs;
}
