import express from 'express';
import { spawn } from 'child_process';
import { createCanvas, Image } from 'canvas';

const app = express();
const algorithms = ['ARIMA', 'LSTM', 'GRU', 'CNN', 'LR_NN'];

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
  res.send(`
    <html>
      <head>
        <title>Crypto Price Prediction</title>
         <link rel="stylesheet" type="text/css" href="/style.css">
      </head>
      <body>
      <header>   
        <img src="/img/logo.png" alt="Logo" class="logo">
      </header>     
        <form class ="form" action="/predict" method="post">
          <label class = "label" for="cryptocurrency">Select cryptocurrency:</label>
          <select class = "select" name="cryptocurrency" id="cryptocurrency">
            <option value="BTC">Bitcoin</option>
            <option value="ETH">Ethereum</option>
            <option value="XRP">XRP</option>
            <option value="DOGE">Dogecoin</option>
          </select>
          <br>
          <label class = "label"  for="algorithm">Select algorithm:</label>
          <select class = "select" name="algorithm" id="algorithm">
            ${algorithms.map(algo => `<option value="${algo}">${algo}</option>`).join('')}
          </select>
          <br>
          <input class = "submit" type="submit" value="Submit">
        </form>
      </body>
    </html>
  `);
});


app.post('/predict', (req, res) => {
    const cryptocurrency = req.body.cryptocurrency;
    const algorithm = req.body.algorithm;
  
    const pythonProcess = spawn('python', [`${algorithm}.py`, cryptocurrency]);
  
    pythonProcess.on('close', (code) => {
      // Create a canvas element
      const canvas = createCanvas(800, 600);
      const context = canvas.getContext('2d');
  
      const img = new Image();
      img.onload = () => {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
        const imgData = canvas.toDataURL();
        res.send(`<div><img src="${imgData}" alt="Crypto price prediction" /></div>`);
      };
      img.src = `${cryptocurrency}_plot.png`;
    });
  });

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});

