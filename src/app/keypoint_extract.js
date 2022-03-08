
// const {dialog} = require('electron').remote;
// const fs = require('fs');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {

console.log(results.multiHandLandmarks)
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
 alert("something is wrong")
 }
 }).then(jsonResponse=>{
 
 // Log the response data in the console
 console.log(jsonResponse)
 } 
 ).catch((err) => console.error(err));

canvasCtx.save();
canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
canvasCtx.drawImage(
    results.image, 0, 0, canvasElement.width, canvasElement.height);
if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                    {color: '#00FF00', lineWidth: 5});
    drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
    }
}
canvasCtx.restore();
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