import ReactDOM from 'react-dom';
import React, { Component } from 'react';
// const electron = window.require('electron');
// const remote = electron.remote
// const {BrowserWindow} = remote
const Webcam = require('react-webcam');


function App() {

    return (
        <div>
            <h1>Hello World</h1>    
            <Webcam />
        </div>        
    )
}

ReactDOM.render(<App />, document.getElementById('root'))
