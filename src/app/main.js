const { app, BrowserWindow, Menu, Tray} = require('electron')

const nativeImage = require('electron').nativeImage;
const icon = nativeImage.createFromPath('icon@2x.png')
const trayicon=nativeImage.createFromPath('icon.png')
app.dock.setIcon(icon);

let win = null;
let tray = null;

function createTray() {
  tray = new Tray(trayicon);
  tray.setToolTip('ASL');
  tray.on('click', event => {
    showWindow();

    // Show devtools when command clicked
    if (win.isVisible() && process.defaultApp && event.metaKey) {
      win.openDevTools({ mode: 'detach' });
    }
  });

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Quit', role:'quit' },
    { label: 'Setting', role:'editMenu' },
  ])
  tray.setContextMenu(contextMenu)

}

// const toggleWindow = () => {
//   if (win.isVisible()) {
//     win.hide();
//   } else {
//     showWindow();
//   }
// };

const showWindow = () => {
  const position = getWindowPosition();
  win.setPosition(position.x, position.y, false);
  win.show();
  win.setVisibleOnAllWorkspaces(true);
  win.focus();
  win.setVisibleOnAllWorkspaces(false);
  
};


const getWindowPosition = () => {
  const windowBounds = win.getBounds();
  const trayBounds = tray.getBounds();

  // Center window horizontally below the tray icon
  const x = Math.round(
    trayBounds.x + trayBounds.width / 2 - windowBounds.width / 2
  );

  // Position window 4 pixels vertically below the tray icon
  const y = Math.round(trayBounds.y + trayBounds.height);

  return { x, y };
};



function createWindow () {
    win = new BrowserWindow({
    width: 300, // fit for 2 video , should be change to 500 then 
    height: 300,
    webPreferences: {
      nodeIntegration: true,
      // enableRemoteModule:true,
    },
     
  })
  win.center();
  //toggleWindow(); //Only Click the tray , win show
  win.loadFile('index.html');
  win.resizable = false// Make the window cannot resize
 
  //win.webContents.openDevTools(); // could use opedev tools to adjust
  //win.loadURL("http://localhost:3000")

}



app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})

app.whenReady().then(() => {
    createWindow();
    createTray();
  
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})




