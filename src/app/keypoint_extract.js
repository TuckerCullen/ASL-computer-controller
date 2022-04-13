// const {dialog} = require('electron').remote;
// const fs = require('fs');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
var prevPoints = null;

function onResults(results) {
console.log(results.poseLandmarks)
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
     console.log("ok");
    return res.json()
 }
 else{
    console.log("no hand detected")
 }
 }).then(jsonResponse=>{
 // Log the response data in the console
 console.log(jsonResponse)
 } 
 ).catch((err) => console.error(err));

 canvasCtx.save();
 canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
 // Only overwrite existing pixels.
 canvasCtx.globalCompositeOperation = 'source-in';
 canvasCtx.fillStyle = '#00FF00';
 canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
 // Only overwrite missing pixels.
 canvasCtx.globalCompositeOperation = 'destination-atop';
 canvasCtx.drawImage(
     results.image, 0, 0, canvasElement.width, canvasElement.height);
 canvasCtx.globalCompositeOperation = 'source-over';
 drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
                {color: '#00FF00', lineWidth: 4});
 drawLandmarks(canvasCtx, results.poseLandmarks,
               {color: '#FF0000', lineWidth: 2});
 drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
                {color: '#CC0000', lineWidth: 5});
 drawLandmarks(canvasCtx, results.leftHandLandmarks,
               {color: '#00FF00', lineWidth: 2});
 drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
                {color: '#00CC00', lineWidth: 5});
 drawLandmarks(canvasCtx, results.rightHandLandmarks,
               {color: '#FF0000', lineWidth: 2});
 canvasCtx.restore();

    if (results.poseLandmarks != null && results.poseLandmarks.length > 0) {
        prevPoints = results.poseLandmarks;
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
width: 300,
height: 168
});
camera.start();