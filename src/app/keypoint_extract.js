
// const {dialog} = require('electron').remote;
// const fs = require('fs');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
var prevPoints = null;

function onResults(results) {

// console.log(results.multiHandLandmarks)
// var multi_hand = JSON.stringify(results.multiHandLandmarks);
// fs.writeFile('myjsonfile.json', multi_hand, 'utf8', callback);
fetch("http://127.0.0.1:5000/receiver", 
 {
 method: 'POST',
 headers: {
 'Content-type': 'application/json',
 'Accept': 'application/json'
 },
 // Strigify the payload into JSON:
 body:JSON.stringify(results.multiHandLandmarks)}).then(res=>{
 if(res.ok){
 return res.json()
 }else{
     console.log("no hand detected")
//  alert("something is wrong")
//  console.log(res)
 }
 }).then(jsonResponse=>{
 
 // Log the response data in the console
//  console.log("KEYPOINT: ", jsonResponse)
 } 
 ).catch((err) => console.error(err));

canvasCtx.save();
canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
canvasCtx.drawImage(
    results.image, 0, 0, canvasElement.width, canvasElement.height);
if (results.multiHandLandmarks) {
    // console.log("prev", prevPoints);
    if (checkStablize(prevPoints, results.multiHandLandmarks)) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                            {color: '#00FF00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
            }
    } else {
        for (const landmarks of prev) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                            {color: '#00FF00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
            }
    }

}
canvasCtx.restore();

if (results.multiHandLandmarks.length > 0) {
    prevPoints = results.multiHandLandmarks;
}

}

function checkStablize(prev, cur) {

    if (prev == null || cur == null) {
        console.log("STABILIZE input null")
        return true
    }
    prev = prev[0]
    cur = cur[0]

    var totalDist = 0

    const distThreshold = .1

    try { 
        for (let i = 0; i < prev.length; i++) {
            var xDelta = prev[i]["x"] - cur[i]["x"]
            var yDelta = prev[i]["y"] - cur[i]["y"]
            var zDelta = prev[i]["z"] - cur[i]["z"]
    
            totalDist += xDelta * xDelta + yDelta * yDelta + zDelta * zDelta
        }
    

        if (totalDist < distThreshold) {
            console.log("STABILIZED");
            return False
        }
        console.log("DIST", totalDist);

        return true
    }
    finally {
        return true
    }
}

const hands = new Hands({locateFile: (file) => {
return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
maxNumHands: 2,
modelComplexity: 1,
minDetectionConfidence: 0.5,
minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
onFrame: async () => {
    await hands.send({image: videoElement});
},
width: 1280,
height: 720
});
camera.start();