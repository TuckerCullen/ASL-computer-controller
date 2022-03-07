import React from 'react';
import ReactDOM from 'react-dom';
// import React, { Component } from 'react';
// import Webcam from "react-webcam";
// const electron = window.require('electron');
// const remote = electron.remote
// const {BrowserWindow} = remote
const Webcam = require('react-webcam');


function App() {
    //const Webcam = require('react-webcam');
    return (
        <div> 
            <Webcam
            height={100+"%"}
            width={100+"%"}
            />
        </div>        
    )
}

ReactDOM.render(<App />, document.getElementById('Video'))

// var startWords = ["S"]


// function startWebcam(){
//   // Setup webcam
  
// }

// function startCommand(){
//   // start Command input

// }

// function stopCommand(){
//   //stop Command input 

// }

// function showCommand(){

// }

// function ExcuteCommand(){
  
// }