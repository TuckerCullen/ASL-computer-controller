
// const {dialog} = require('electron').remote;
// const fs = require('fs');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
var prevPoints = null;

function onResults(results) {

// console.log(results.poseLandmarks)
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
 //[results.poseLandmarks, results.leftHandLandmarks, results.rightHandLandmarks]
 body:JSON.stringify(results)}).then(res=>{
 if(res.ok){
 return res.json()
 }else{
 console.log("no hand detected")
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

if (results.multiHandLandmarks != null && results.multiHandLandmarks.length > 0) {
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

const holistic = new Holistic({locateFile: (file) => {
return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});
holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
holistic.onResults(onResults);

const camera = new Camera(videoElement, {
onFrame: async () => {
    await holistic.send({image: videoElement});
},
width: 250,
height: 140
});
camera.start();