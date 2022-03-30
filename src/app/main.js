const { app, BrowserWindow } = require('electron')


function createWindow () {
  const win = new BrowserWindow({
    width: 500,
    height: 300,
    webPreferences: {
      nodeIntegration: true,
      // enableRemoteModule:true,
    }
    
  })
  win.loadFile('index.html');
  win.webContents.openDevTools(); // could use opedev tools to adjustxw
  //win.loadURL("http://localhost:3000")
  win.webContents.openDevTools();
}


app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})

app.whenReady().then(() => {
    createWindow()
  
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})


function startWebcam(){
  // Setup webcam
  
}


