const { app, BrowserWindow ,ipcMain } = require('electron');
// var main=null;



function createWindow () {
  const win = new BrowserWindow({
    width: 500,
    height: 350,
    webPreferences: {
    nodeIntegration: true, // allow render.js use node.js
    enableRemoteModule:true,
    contextIsolation:false,
    }
    
  })
  
  win.loadFile('index.html');
  win.webContents.openDevTools(); // could use opedev tools to adjust
  //win.loadURL("http://localhost:3000")


}




app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})

app.whenReady().then(() => {
    createWindow();
    
   
  
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();

    })
    
});

ipcMain.on('start',cmd =>{
  console.log(cmd)
  

})