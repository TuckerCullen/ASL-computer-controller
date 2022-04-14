// const {dialog} = require('electron').remote;
// const fs = require('fs');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
var prevPoints = null;
var prevPoints_dir = null;
var if_not_stabilize = true;


const startBut = document.querySelector('#start')
const stopBut = document.querySelector('#stop')
const statusText = document.getElementById("status-text");
const command = document.querySelector('#Interaction-text');
const recieved_prediction = false;
console.log(start)


function onResults(results) {
    var checkstatus = true;
    
    if(prevPoints_dir == 'left'){
        checkstatus = checkStablize(prevPoints, results.leftHandLandmarks);
        console.log(checkStablize(prevPoints, results.leftHandLandmarks))
    }else if(prevPoints_dir == 'right'){
        checkstatus= checkStablize(prevPoints, results.rightHandLandmarks);
        console.log(checkStablize(prevPoints, results.leftHandLandmarks))
    }
    
    

    
    if(checkstatus){
        
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
        if_not_stabilize=true;

    } 
    else if (if_not_stabilize) {
        console.log(if_not_stabilize)
        // start prediction
        fetch("http://127.0.0.1:5000/predict")

        fetch("http://127.0.0.1:5000/sender")
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            console.log('GET response:');
            console.log(text.Prediction);
            document.getElementById("Status").style.background='#33C2FF';
            statusText.innerText = "Ready";
            command.style.color='#33C2FF';
            startBut.style.display="none";
            command.innerText =text.Prediction;
            //recieved_prediction=true;

            setTimeout(() => {
                // Back to origin status 
            canvasElement.style.display="none";
            videoElement.style.display="block"; 

            document.getElementById("Status").style.background='#FF6633';
            statusText.innerText = "Not Ready";
            command.style.color='#FF6633';
            command.innerText ="Hello,Sign"; 
            startBut.style.display="block";
            console.log("test1");
              }, 10000)
            })
        
        if_not_stabilize=false;
       }
          
        
    

    if (results.leftHandLandmarks != null && results.leftHandLandmarks.length > 0) {
        prevPoints = results.leftHandLandmarks;
        prevPoints_dir = "left";
    }else if(results.rightHandLandmarks != null && results.rightHandLandmarks.length > 0){
        prevPoints = results.rightHandLandmarks;
        prevPoints_dir = "right";
    }
}

function checkStablize(prev, cur) {

    if (prev == null || cur == null) {
        console.log("STABILIZE input null")
        return true
    }

    var totalDist = 0

    const distThreshold = .1

        for (let i = 0; i < prev.length; i++) {
            var xDelta = prev[i]["x"] - cur[i]["x"]
            var yDelta = prev[i]["y"] - cur[i]["y"]
            var zDelta = prev[i]["z"] - cur[i]["z"]
    
            totalDist += xDelta * xDelta + yDelta * yDelta + zDelta * zDelta
        }
    

        if (totalDist < distThreshold) {
            console.log("STABILIZED");
            return false
        }
        console.log("DIST", totalDist);

        return true
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

  
  
  const camera = new Camera(videoElement, {
    onFrame: async () => {
        await holistic.send({image: videoElement});
    },
    width: 300,
    height: 168
    });
    camera.start();




    // START RECIEVE COMMAND 
startBut.onclick = function(){
    holistic.onResults(onResults);
    // camera.onFrame=async () => {
    //     await holistic.send({image: videoElement});
    // }
   
    canvasElement.style.display="block";
    videoElement.style.display="none"; 

    document.getElementById("Status").style.background='#00C897';
    statusText.innerText = "Signing";
    command.style.color='#00C897';
    command.innerText ="Translating.... "; 
    startBut.style.display="none";
    console.log("test1");
}



//  PrdictResult() {
//         // var multi_hand = JSON.stringify(results.multiHandLandmarks);
//         // fs.writeFile('myjsonfile.json', multi_hand, 'utf8', callback);
//         const res= fetch("http://127.0.0.1:5000/sender", 
//          { method: 'GET'}
         
//  }
// STOP RECIEVE COMMAND, Back to 

stopBut.onclick = function(){

// Commnad use to control system 
// Wait for prediction result
    
fetch("http://127.0.0.1:5000/sender")
.then(function (response) {
    return response.json();
}).then(function (text) {
    console.log('GET response:');
    console.log(text.Prediction);
    document.getElementById("Status").style.background='#33C2FF';
    statusText.innerText = "Ready";
    command.style.color='#33C2FF';
    stopBut.style.background='#33C2FF';
    command.innerText =text.Prediction;
});
// Back to origin status 
    document.getElementById("Status").style.background='#FF6633';
    statusText.innerText = "Not ready";
    command.style.color='#FF6633';
    command.innerText ="Hello, Sign"
    stopBut.style.display="none";
    startBut.style.display="block";
}


