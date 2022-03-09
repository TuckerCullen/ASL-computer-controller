import React from 'react';
import ReactDOM from 'react-dom';

// import React, { Component } from 'react';
// import Webcam from "react-webcam";
// const electron = window.require('electron');
// const remote = electron.remote
// const {BrowserWindow} = remote


function Video() {
   const Webcam = require('react-webcam');
    
    return (
        <div> 
            <Webcam
            height={100+"%"}
            width={100+"%"}
            />
        </div>      
    )
}


ReactDOM.render(<Video />, document.getElementById('Video'))

