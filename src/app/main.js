
const { app, BrowserWindow } = require('electron')


function createWindow () {
  const win = new BrowserWindow({
    width: 500,
    height: 500,
    webPreferences: {
      nodeIntegration: true,
      // enableRemoteModule:true,
    }
    
  })
  win.loadFile('index.html');
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


