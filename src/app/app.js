
// All function 



const startBut = document.querySelector('#start')
const stopBut = document.querySelector('#stop')
const statusText = document.getElementById("status-text");
const command = document.querySelector('#Interaction-text');



// START RECIEVE COMMAND 
startBut.onclick = function(){
    document.getElementById("Status").style.background='#00C897';
    statusText.innerText = "Wait for command";
    command.style.color='#00C897';
    command.innerText ="Translating ..."; 
    startBut.style.display="none";
    stopBut.style.display="block";
    // console.log(PrdictResult());
    console.log("test1");
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

// Back to origin status 
    document.getElementById("Status").style.background='#FF6633';
    statusText.innerText = "Not ready";
    command.style.color='#FF6633';
    command.innerText ="Hello, Sign"
    stopBut.style.display="none";
    startBut.style.display="block";
}







