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
 * @fileoverview
 * Tic-Tac-Toe Neural Network Model and Prediction Functions
 *
 * @description
 * This script defines a simple neural network model using TensorFlow.js for making predictions
 * in a Tic-Tac-Toe game. It includes functions for training the neural network and making predictions
 * based on the current game state.
 *
 * @author Your Name
 * @version 1.0
 */

// Create a sequential neural network model for Tic-Tac-Toe prediction
const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [9], units: 128, activation: 'relu' }),
    tf.layers.dense({ units: 64, activation: 'relu' }),
    tf.layers.dense({ units: 9, activation: 'softmax' }),
  ],
});

/**
 * Train the neural network using provided training data.
 * @async
 */
async function trainNeuralNetwork() {
  // Prepare input and output tensors for training
  const xs = tf.tensor2d(trainingData.input, [trainingData.input.length, 9]);
  const ys = tf.oneHot(tf.tensor1d(trainingData.output).toInt(), 9);

  // Compile the model with optimizer, loss function, and metrics
  await model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Train the model with input and output tensors for a specified number of epochs
  await model.fit(xs, ys, { epochs: 10 });

  // Dispose of input and output tensors to free up memory
  xs.dispose();
  ys.dispose();
}

/**
 * Make a prediction using the trained neural network for the current game state.
 * @returns {number} - The predicted move index.
 */
function neuralNetworkPrediction() {
  // Convert the current game board to a tensor
  const inputTensor = tf.tensor2d([origBoard], [1, 9]);

  // Make a prediction using the neural network
  const prediction = model.predict(inputTensor);

  // Get the index of the predicted move with the highest probability
  const predictedMove = tf.argMax(prediction, 1).dataSync()[0];

  // Dispose of input tensor and prediction tensor to free up memory
  inputTensor.dispose();
  prediction.dispose();

  // Return the predicted move index
  return predictedMove;
}
