// All function 

const start = document.querySelector('#start')
const statusText = document.getElementById("status-text");
console.log(start)


start.onclick = function(){
    document.getElementById("Status").style.background='#00C897'
    document.querySelector('#Interaction-text').style.color='#00C897'
    statusText.innerText = "Wait for command";
    start.style.display = 'none';
}



// need trigger sign 
//start and end prediction
// command control