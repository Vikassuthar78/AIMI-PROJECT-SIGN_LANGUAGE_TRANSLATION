async function fetchPrediction() {
  const response = await fetch('/get_prediction');
  const data = await response.json();
  document.getElementById('gesture').textContent = data.gesture;
}

setInterval(fetchPrediction, 1000); // Update every second
