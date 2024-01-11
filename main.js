/**
 * @fileoverview
 * Library for predicting a move based on algo
 * Provides functions for game initialization, player moves, Algo base AI predictions,
 * and determining the game outcome.
 *
 * @author RSK
 * @version 1.0
 */

const huPlayer = 'O';
const aiPlayer = 'X';
const winCombo = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [6, 4, 2],
];

// Array to represent the game board
var origBoard;

// Select all cells with class 'cell'
const cell = document.querySelectorAll('.cell');

// Initialize the game
startGame();

/**
 * Handle a cell click event.
 * @param {Event} square - The clicked cell.
 */
function turnClick(square) {
  if (typeof origBoard[square.target.id] === 'number') {
    turn(square.target.id, huPlayer);
    //trainNeuralNetwork();
    if (!checkTie()) turn(bestSpot(), aiPlayer);
  }
}

/**
 * Update the game board and check for a winner after each move.
 * @param {number} squareId - The id of the clicked cell.
 * @param {string} player - The symbol of the current player.
 */
function turn(squareId, player) {
  origBoard[squareId] = player;
  document.getElementById(squareId).innerText = player;

  // Check for a winner after each move
  let gameWon = checkWin(origBoard, player);
  if (gameWon) gameOver(gameWon);
}

/**
 * Check if a player has won the game.
 * @param {Array} board - The current game board.
 * @param {string} player - The symbol of the current player.
 * @returns {Object|null} - An object containing the winning index and player, or null if no winner.
 */
function checkWin(board, player) {
  let plays = board.reduce((a, e, i) => (e === player ? a.concat(i) : a), []);
  let gameWon = null;
  for (let [index, win] of winCombo.entries()) {
    if (win.every((elem) => plays.indexOf(elem) > -1)) {
      gameWon = { index: index, player: player };
      break;
    }
  }
  return gameWon;
}

/**
 * Handle the end of the game, update UI and display the winner.
 * @param {Object} gameWon - Object containing the winning index and player.
 */
function gameOver(gameWon) {
  for (let index of winCombo[gameWon.index]) {
    document.getElementById(index).style.backgroundColor = gameWon.player == huPlayer ? 'blue' : 'red';
  }
  for (var i = 0; i < cell.length; i++) {
    cell[i].removeEventListener('click', turnClick, false);
  }
  declareWinner(gameWon.player == huPlayer ? 'You win!' : 'You lose.');
}

/**
 * Display the winner and endgame message.
 * @param {string} who - The winner or game outcome message.
 */
function declareWinner(who) {
  document.querySelector('.endgame').style.display = 'block';
  document.querySelector('.endgame .text').innerText = who;
}

/**
 * Get an array of empty squares on the game board.
 * @returns {Array} - Array containing the indices of empty squares.
 */
function emptySquares() {
  return origBoard.filter((s) => typeof s == 'number');
}

/**
 * Determine the best spot for the AI to make a move using the minimax algorithm.
 * @returns {number} - The index of the best move.
 */
function bestSpot() {
  return minimax(origBoard, aiPlayer).index;
}

/**
 * Check if the game is a tie.
 * @returns {boolean} - True if the game is a tie, false otherwise.
 */
function checkTie() {
  if (emptySquares().length === 0) {
    for (var i = 0; i < cell.length; i++) {
      cell[i].style.backgroundColor = 'green';
      cell[i].removeEventListener('click', turnClick, false);
    }
    declareWinner('Tie Game!');
    return true;
  }
  return false;
}

/**
 * Neural Network Model Configuration
 * @type {tf.Sequential}
 */
const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [9], units: 128, activation: 'relu' }),
    tf.layers.dense({ units: 64, activation: 'relu' }),
    tf.layers.dense({ units: 9, activation: 'softmax' }),
  ],
});

/**
 * Train the Neural Network using provided training data.
 * @async
 */
async function trainNeuralNetwork() {
  // Convert input and output data to TensorFlow tensors
  const xs = tf.tensor2d(trainingData.input, [trainingData.input.length, 9]);
  const ys = tf.oneHot(tf.tensor1d(trainingData.output).toInt(), 9);

  // Compile the model
  await model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Fit the model to the training data
  await model.fit(xs, ys, { epochs: 10 });

  // Dispose of the tensors to free up memory
  xs.dispose();
  ys.dispose();
}

/**
 * Predict the next move using the trained neural network.
 * @returns {number} - Index of the predicted move.
 */
function neuralNetworkPrediction() {
  // Convert the current game board to a TensorFlow tensor
  const inputTensor = tf.tensor2d([origBoard], [1, 9]);

  // Make a prediction using the trained model
  const prediction = model.predict(inputTensor);

  // Get the index of the move with the highest probability
  const predictedMove = tf.argMax(prediction, 1).dataSync()[0];

  // Dispose of the input tensor and prediction tensor to free up memory
  inputTensor.dispose();
  prediction.dispose();

  return predictedMove;
}

for (let i = 0; i < numWorkers; i++) {
  const worker = new Worker(
    URL.createObjectURL(
      new Blob(
        [
          `
        onmessage = function(event) {
            const { Perceptron, trainingData } = event.data;

            // Create a new Perceptron instance
            const perceptron = new Perceptron(2);

            // Train the perceptron with a subset of the training data
            for (let epoch = 0; epoch < 1000; epoch++) {
                for (const data of trainingData) {
                    perceptron.train(data.inputs, data.target, 0.1);
                }
            }

            // Send the updated weights and bias back to the main thread
            postMessage({ weights: perceptron.weights, bias: perceptron.bias });
        }
    `,
        ],
        { type: 'application/javascript' }
      )
    )
  );

  workers.push(worker);

  worker.onmessage = function (event) {
    // Update perceptron weights and bias with the results from each worker
    const { weights, bias } = event.data;
    perceptron.weights = weights;
    perceptron.bias = bias;

    // Display current perceptron state
    console.log('Epoch:', i + 1, 'Weights:', perceptron.weights, 'Bias:', perceptron.bias);
  };

  worker.postMessage({
    Perceptron,
    trainingData: trainingData.slice(i * (trainingData.length / numWorkers), (i + 1) * (trainingData.length / numWorkers)),
  });
}

/**
 * Save game data to a file using the File System Access API.
 * @async
 * @param {Object} gameData - The game data to be saved.
 */
async function saveGameDataToFile(gameData) {
  try {
    // Prompt the user to select a location to save the file
    const fileHandle = await window.showSaveFilePicker();

    // Create a writable stream to the selected file
    const writableStream = await fileHandle.createWritable();

    // Write the JSON-stringified game data to the file
    await writableStream.write(JSON.stringify(gameData));

    // Close the writable stream
    await writableStream.close();

    console.log('Game data saved successfully.');
  } catch (error) {
    console.error('Error saving game data:', error);
  }
}

// Example usage
const gameData = {
  /* Your game data here */
};
//saveGameDataToFile(gameData);

function gameOver(gameWon) {
  for (let index of winCombo[gameWon.index]) {
    document.getElementById(index).style.backgroundColor = gameWon.player == huPlayer ? 'blue' : 'red';
  }
  for (var i = 0; i < cell.length; i++) {
    cell[i].removeEventListener('click', turnClick, false);
  }

  // Check if the player has won
  if (gameWon.player == huPlayer) {
    // Player won - send email
    sendWinNotificationEmail();
  }

  declareWinner(gameWon.player == huPlayer ? 'You win!' : 'You lose.');
}

function sendWinNotificationEmail() {
  // Assuming you have a server-side endpoint to send emails
  const emailEndpoint = '/send-email'; // Replace with your actual endpoint

  // You may need to include additional data in the email, like the player's details
  const emailData = {
    to: 'recipient@example.com', // Replace with the recipient's email address
    subject: 'Congratulations! You won the game!',
    body: 'You are the champion! Play again and enjoy.',
  };

  // Use fetch or another method to send a POST request to the server-side endpoint
  fetch(emailEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(emailData),
  })
    .then((response) => response.json())
    .then((data) => console.log('Email sent:', data))
    .catch((error) => console.error('Error sending email:', error));
}
