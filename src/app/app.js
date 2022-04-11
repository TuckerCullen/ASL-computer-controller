
// All function 



const startBut = document.querySelector('#start')
const stopBut = document.querySelector('#stop')
const statusText = document.getElementById("status-text");
const command = document.querySelector('#Interaction-text');
console.log(start)



// START RECIEVE COMMAND 
start.onclick = function(){
    document.getElementById("Status").style.background='#00C897';
    statusText.innerText = "Wait for command";
    command.style.color='#00C897';
    startBut.style.display="none";
    stopBut.style.display="block";
    startPredict();

}

// startPredict(){
//    Predict

//    command.innerText =requestAnimationFrame()
// }

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








