let handPose;

console.log("Hello");
function preload() {
  handPose = ml5.handPose();
}

let video;

function setup() {
  createCanvas(640, 480); // Create the video and hide it
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  handPose.detectStart(video, gotHands);
}

let hands = [];

// Callback function for when handPose outputs data
function gotHands(results) {
  // Save the output to the hands variable
  hands = results;
}

function draw() {
  image(video, 0, 0, width, height);

  for (let i = 0; i < hands.length; i++) {
    let hand = hands[i];

    for (let j = 0; j < hand.keypoints.length; j++) {
      let keypoint = hand.keypoints[j];

      fill(0, 255, 0);
      noStroke();
      circle(keypoint.x, keypoint.y, 10);

      // Number label
      fill(255);
      textSize(14);
      text(j, keypoint.x + 5, keypoint.y - 5);
    }
  }
}
